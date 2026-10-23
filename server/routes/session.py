"""The caller's session: role, guest quota and retention."""

from fastapi import APIRouter

from ..schemas import SessionResponse
from ..sessions import CurrentSession, session_info

router = APIRouter(prefix="/v1", tags=["session"])


@router.get(  # type: ignore[misc]
    "/session",
    response_model=SessionResponse,
    summary="Current session",
)
async def get_session(session: CurrentSession) -> SessionResponse:
    """Role, analyses used in the last hour (limited for guests), retention and the model-download flag.

    The first request without a valid ``dt4lc_session`` cookie gets one; every request is a
    guest until sign-in exists, unless the server sets ``DEV_ROLE``.
    """
    return SessionResponse.model_validate(session_info(session))  # type: ignore[no-any-return]
