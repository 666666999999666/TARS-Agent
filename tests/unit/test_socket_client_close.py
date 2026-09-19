"""Controlled StreamReaderProtocol close failures; no OS processes or Docker."""
from __future__ import annotations

import asyncio
import gc
import json

import pytest

from tars_agent.core.transport.socket_client import IpcDisconnectedError, IpcError, SocketClient


class ControlledTransport(asyncio.Transport):
    def __init__(self, protocol, *, hold_close=False, close_error=None):
        super().__init__()
        self.protocol = protocol
        self.hold_close = hold_close
        self.close_error = close_error
        self.close_called = asyncio.Event()
        self.written = asyncio.Event()
        self.frames = []
        self.closing = False
        self.finished = False
        self.aborted = False

    def get_extra_info(self, name, default=None):
        return default

    def is_closing(self):
        return self.closing

    def write(self, data):
        self.frames.append(json.loads(data))
        self.written.set()

    def close(self):
        self.close_called.set()
        if self.close_error is not None:
            raise self.close_error
        self.closing = True
        if not self.hold_close:
            self.finish()

    def finish(self, error=None):
        if not self.finished:
            self.finished = True
            self.closing = True
            # Real asyncio protocol: the same exception reaches reader and wait_closed.
            self.protocol.connection_lost(error)

    def abort(self):
        self.aborted = True
        self.finish()


class Peer:
    def __init__(self, **kwargs):
        reader = asyncio.StreamReader()
        self.protocol = asyncio.StreamReaderProtocol(reader)
        self.transport = ControlledTransport(self.protocol, **kwargs)
        self.protocol.connection_made(self.transport)
        writer = asyncio.StreamWriter(self.transport, self.protocol, reader, asyncio.get_running_loop())
        self.client = SocketClient("127.0.0.1", 1)
        self.client._reader, self.client._writer = reader, writer
        self.tasks = []

    def task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def pending(self):
        loop = self.task(self.client.run_event_loop())
        request = self.task(self.client.send_command("run.get", {"run_id": "controlled"}))
        await self.transport.written.wait()
        return loop, request


@pytest.fixture
async def peers():
    created = []

    def make(**kwargs):
        peer = Peer(**kwargs)
        created.append(peer)
        return peer

    yield make
    # Failure cleanup cannot establish a passing assertion: tests check completion first.
    for peer in created:
        peer.transport.close_error = None
        peer.transport.finish()
        for task in peer.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*peer.tasks, return_exceptions=True)
        closed = peer.protocol._get_close_waiter(None)
        if closed.done() and not closed.cancelled():
            closed.exception()


async def completed(*tasks):
    _, pending = await asyncio.wait(tasks, timeout=2)
    assert not pending, "client tasks must finish before test cleanup; no timeout cancellation"


async def test_normal_and_repeated_close_finish_reader_and_fail_pending(peers):
    peer = peers()
    loop, request = await peer.pending()
    await peer.client.close()
    await peer.client.close()
    await completed(loop, request)
    assert loop.result() is None
    with pytest.raises(IpcDisconnectedError, match="client connection closed"):
        request.result()
    assert not peer.client._pending
    with pytest.raises(IpcDisconnectedError):
        await peer.client.send_command("run.get", {})


@pytest.mark.parametrize("error", [ConnectionResetError("peer reset"), BrokenPipeError("peer closed")])
async def test_peer_disconnect_is_handled_by_reader_then_close_waiter(peers, error):
    peer = peers()
    loop, request = await peer.pending()
    peer.transport.finish(error)
    await completed(loop, request)
    assert loop.result() is None  # The read loop already handled the disconnect.
    with pytest.raises(IpcDisconnectedError):
        request.result()
    # Demonstrate the exact source: asyncio's protocol close waiter retains this error.
    assert peer.protocol._get_close_waiter(None).exception() is error
    await peer.client.close()
    await peer.client.close()
    assert not peer.client._pending


async def test_cancel_close_aborts_stalled_transport_and_preserves_cancellation(peers):
    peer = peers(hold_close=True)
    loop, request = await peer.pending()
    closing = peer.task(peer.client.close())
    await peer.transport.close_called.wait()
    closing.cancel()
    await completed(closing)
    with pytest.raises(asyncio.CancelledError):
        closing.result()
    assert peer.transport.aborted
    await completed(loop, request)
    assert loop.result() is None
    with pytest.raises(IpcDisconnectedError):
        request.result()
    # Cancellation must not poison the next close via a cancelled shared close waiter.
    await peer.client.close()


async def test_close_timeout_releases_transport_without_hanging_reader(peers):
    peer = peers(hold_close=True)
    loop, request = await peer.pending()
    closing = peer.task(peer.client.close())
    await completed(closing)
    assert closing.result() is None
    assert peer.transport.aborted
    await completed(loop, request)
    assert loop.result() is None
    with pytest.raises(IpcDisconnectedError):
        request.result()
    await peer.client.close()


@pytest.mark.parametrize("error", [RuntimeError("unexpected waiter failure"), OSError(64, "not a reset exception")])
async def test_unexpected_wait_closed_error_still_propagates(peers, error):
    peer = peers()
    peer.protocol._get_close_waiter(None).set_exception(error)
    with pytest.raises(type(error)) as caught:
        await peer.client.close()
    assert caught.value is error
    assert peer.transport.closing


@pytest.mark.parametrize("error", [ConnectionResetError("close call failure"), RuntimeError("close call failure")])
async def test_errors_from_close_itself_are_not_misclassified_as_waiter_disconnect(peers, error):
    peer = peers(close_error=error)
    request = peer.task(peer.client.send_command("run.get", {}))
    await peer.transport.written.wait()
    with pytest.raises(type(error)) as caught:
        await peer.client.close()
    assert caught.value is error
    await completed(request)
    with pytest.raises(IpcDisconnectedError):
        request.result()


async def test_rpc_business_failure_remains_failure_after_disconnect_cleanup(peers):
    peer = peers()
    loop, request = await peer.pending()
    frame = peer.transport.frames[0]
    await peer.client._dispatch(json.dumps({"jsonrpc": "2.0", "id": frame["id"],
        "error": {"code": -32033, "message": "cleanup still pending", "data": {"cancellation_requested": True}},
    }).encode())
    await completed(request)
    original = request.exception()
    assert isinstance(original, IpcError) and original.code == -32033
    peer.transport.finish(ConnectionResetError("peer exited after reporting failure"))
    await peer.client.close()
    await completed(loop)
    assert loop.result() is None
    assert request.exception() is original
    assert original.data == {"cancellation_requested": True}


async def test_business_assertion_is_not_replaced_by_expected_close_error(peers):
    peer = peers()
    peer.transport.finish(ConnectionResetError("peer lost"))
    with pytest.raises(AssertionError, match="business assertion remains"):
        try:
            raise AssertionError("business assertion remains")
        finally:
            await peer.client.close()


async def test_cancelled_request_is_not_turned_into_success_by_close(peers):
    peer = peers()
    loop, request = await peer.pending()
    request.cancel()
    await completed(request)
    peer.transport.finish(ConnectionResetError("peer lost"))
    await peer.client.close()
    await completed(loop)
    assert loop.result() is None
    assert request.cancelled() and not peer.client._pending


@pytest.mark.parametrize("cancel_request", [False, True])
async def test_close_while_request_waits_for_write_lock_leaves_no_unretrieved_error(peers, cancel_request):
    class ObservedLock(asyncio.Lock):
        attempted = asyncio.Event()

        async def acquire(self):
            self.attempted.set()
            return await super().acquire()

    peer = peers()
    lock = ObservedLock()
    peer.client._write_lock = lock
    await lock.acquire()
    lock.attempted.clear()
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        request = peer.task(peer.client.send_command("run.get", {}))
        await lock.attempted.wait()
        await peer.client.close()
        if cancel_request:
            request.cancel()
        lock.release()
        await completed(request)
        if cancel_request:
            assert request.cancelled()
        else:
            assert isinstance(request.exception(), IpcDisconnectedError)
        assert not peer.transport.frames and not peer.client._pending
        peer.tasks.remove(request)
        del request
        gc.collect()  # Exercise the event loop's actual unhandled-Future diagnostic.
        assert not errors, [error.get("message") for error in errors]
    finally:
        if lock.locked():
            lock.release()
        loop.set_exception_handler(previous)
