"""Single-attempt HTTP transport. No store imports, storage paths, or provider keys."""
import json
import math
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler


class RemoteError(RuntimeError):
    """A remote failure; unknown outcome never authorizes a mutation retry."""
    def __init__(self, code, outcome='unknown', status=None):
        self.code, self.outcome, self.status = code, outcome, status
        super().__init__(f'RAG HTTP {code} (outcome={outcome})')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _constant(_):
    raise ValueError('non-finite JSON')


def _chunks(value):
    return isinstance(value, list) and all(
        isinstance(c, dict) and set(c) == {'text', 'metadata', 'score'}
        and isinstance(c['text'], str) and isinstance(c['metadata'], dict)
        and type(c['score']) in (int, float) and math.isfinite(c['score'])
        for c in value)


def _strings(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _count(value):
    return type(value) is int and value >= 0


class Client:
    """Explicit target, 60s socket timeout and 16MiB response cap by default.

    Timeout is a socket-I/O timeout, not a whole-pipeline deadline. No redirects,
    retries, target discovery side effects, Chroma fallback or secret forwarding.
    Use the enclosing executor's deadline when a wall-clock bound is required.
    """
    def __init__(self, base_url, namespace, space, *, token='', timeout=60,
                 max_response_bytes=16 * 1024 * 1024, max_request_bytes=16 * 1024 * 1024):
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('base_url must be an explicit HTTP(S) service URL')
        if not isinstance(namespace, str) or not namespace:
            raise ValueError('namespace must be explicit')
        if not isinstance(space, dict) or set(space) != {'provider', 'model', 'dimension', 'schema'}:
            raise ValueError('space must be explicit')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError('response limit must be positive')
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError('request limit must be positive')
        self.max_request_bytes = max_request_bytes
        self.base_url, self.namespace = base_url.rstrip('/'), namespace
        self.space, self.token = dict(space), token
        self.timeout, self.max_response_bytes = timeout, max_response_bytes
        self._opener = build_opener(_NoRedirect())

    def _call(self, operation, fields, validate, *, space=True):
        body = {'namespace': self.namespace, **fields}
        if space:
            body['space'] = self.space
        # Serialization failure occurs before any request and is a local error.
        payload = json.dumps(body, allow_nan=False, ensure_ascii=True).encode('utf-8')
        if len(payload) > self.max_request_bytes:
            raise ValueError('request exceeds configured byte limit')
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        request = Request(self.base_url + '/v1/' + operation, data=payload, headers=headers, method='POST')
        try:
            try:
                response = self._opener.open(request, timeout=self.timeout)
            except HTTPError as exc:
                response = exc
            with response:
                status = response.code
                data = response.read(self.max_response_bytes + 1)
                if len(data) > self.max_response_bytes:
                    raise RemoteError('response_too_large', status=status)
                value = json.loads(data, parse_constant=_constant)
                if status != 200:
                    if isinstance(value, dict) and isinstance(value.get('error'), dict):
                        error = value['error']
                        if isinstance(error.get('code'), str) and error.get('outcome') in ('not_started', 'unknown'):
                            raise RemoteError(error['code'], error['outcome'], status)
                    raise RemoteError('invalid_response', status=status)
                if not validate(value):
                    raise RemoteError('invalid_response', status=status)
                return value
        except RemoteError:
            raise
        except Exception:
            # Do not expose provider bodies, URL credentials or transport details.
            raise RemoteError('transport_or_protocol_error') from None

    def inspect(self):
        return self._call('inspect', {}, lambda v: isinstance(v, dict)
                          and v.get('namespace') == self.namespace and type(v.get('exists')) is bool
                          and isinstance(v.get('space'), dict) and isinstance(v.get('settings'), dict)
                          and _strings(v.get('generation_models')) and _count(v.get('total_chunks'))
                          and _strings(v.get('documents')), space=False)

    def create(self):
        return self._call('create', {}, lambda v: isinstance(v, dict) and v.get('namespace') == self.namespace and v.get('created') is True)

    def replace(self, chunks, embeddings, doc_id, metadata=None):
        return self._call('replace', dict(chunks=chunks, embeddings=embeddings, doc_id=doc_id, metadata=metadata), lambda v: v is None)

    def ingest(self, text, doc_id, *, chunk_size, chunk_overlap, strategy='fixed', metadata=None):
        return self._call('ingest', dict(text=text, doc_id=doc_id, metadata=metadata, strategy=strategy,
                          settings=dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap)),
                          lambda v: isinstance(v, dict) and v.get('doc_id') == doc_id and _count(v.get('chunks_stored')))

    def fetch(self, vector, *, top_k, score_threshold):
        return self._call('fetch', dict(vector=vector, settings=dict(top_k=top_k, score_threshold=score_threshold)), _chunks)

    def query(self, question, *, top_k, score_threshold, generation_model):
        return self._call('query', dict(question=question, settings=dict(top_k=top_k, score_threshold=score_threshold, generation_model=generation_model)),
                          lambda v: isinstance(v, dict) and isinstance(v.get('answer'), str)
                          and _strings(v.get('sources')) and _count(v.get('chunk_count')) and _chunks(v.get('chunks')))

    def delete(self, doc_id):
        return self._call('delete', dict(doc_id=doc_id), lambda v: isinstance(v, dict)
                          and v.get('doc_id') == doc_id and _count(v.get('chunks_deleted')))
