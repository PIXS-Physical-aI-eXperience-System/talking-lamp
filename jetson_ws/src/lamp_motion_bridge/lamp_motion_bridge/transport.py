"""Persistent Pi motion transport with correlation and no command replay."""

from lamp_device_bridge.transport import DeviceTransport, TransportError


class MotionTransport(DeviceTransport):
    """Motion wire framing is identical to device request/result framing.

    The Pi motion endpoint never emits server-pushed events; inherited event
    support remains dormant while request correlation, heartbeat, reconnect,
    bounds and fail-pending behavior stay shared with the device transport.
    """


__all__ = ["MotionTransport", "TransportError"]
