from multiprocessing import Event
import socket

import logging
logger = logging.getLogger(__name__)

def serve_forever_process(
    server_fd: int,
    conf: dict,
    # stop_event: multiprocessing.Event,
    stop_event: Event,
    shared_validation_status,
    shared_validation_lock,
    run_validation_scheduler: bool,
    validation_interval: int,
):
    """Worker process entrypoint: accept connections and handle them in-process."""
    server = socket.fromfd(server_fd, socket.AF_UNIX, socket.SOCK_STREAM)
    server.settimeout(1.0)

    from controller.socket_controller.socket_controller import SocketServer

    # Build isolated controller state per process
    srv = SocketServer(
        conf=conf,
        shared_validation_status=shared_validation_status,
        shared_validation_lock=shared_validation_lock,
        run_validation_scheduler=run_validation_scheduler,
    )
    srv.socket_conf = conf.get("socket_connection", {})
    srv.worker_processes = 1  # prevent recursive forking in workers
    srv.running = True
    srv._stop_event = stop_event  # use shared event for exit signal
    srv._mp_stop_event = stop_event
    srv.validation_interval = validation_interval if run_validation_scheduler else 0

    if srv.validation_interval > 0 and srv._run_validation_scheduler:
        srv._start_validation_scheduler()

    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
                with conn:
                    srv.handle_connection_safe(conn)
            except socket.timeout:
                continue
            except OSError as e:
                if stop_event.is_set():
                    break
                logger.error(f"Worker socket error: {e}")
                break
            except KeyboardInterrupt:
                logger.info("Worker received interrupt, stopping...")
                break
            except Exception as e:  # noqa: BLE001
                if not stop_event.is_set():
                    logger.error(f"Worker error: {e}")
    finally:
        stop_event.set()

    server.close()