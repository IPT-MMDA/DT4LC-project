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

    The first request without a valid ``dt4lc_session`` cookie gets one, as a guest. Signing in
    makes the session a member or an operator (``ALLOWED_EMAILS``, ``ALLOWED_DOMAINS``,
    ``OPERATOR_EMAILS``); a server that sets ``DEV_ROLE`` gives every request that role.
    """
    return SessionResponse.model_validate(session_info(session))  # type: ignore[no-any-return]
