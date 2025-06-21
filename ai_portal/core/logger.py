import structlog

def setup_logger():
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    return structlog.get_logger()