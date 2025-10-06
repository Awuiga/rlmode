from app.common.redis_stream import RedisStream
from pydantic import BaseModel


class M(BaseModel):
    a: int
    b: None | dict = None


def test_xadd_cleans_none_and_json(mocker):
    rs = RedisStream("fakeredis://")
    spy = mocker.spy(rs.client, "xadd")
    rs.xadd("t:stream", M(a=1, b={"x": [1, 2]}))
    args = spy.call_args[0]
    kwargs = spy.call_args[1]
    assert args[0] == "t:stream"
    payload = args[1]
    assert "b" in payload and isinstance(payload["b"], str)
    assert payload.get("a") == 1
    assert "maxlen" not in kwargs or kwargs["maxlen"] is None

