"""Read-only USB control adapter for reSpeaker Flex XVF3800 Linear-4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import struct
import time
from typing import Any


XVF_VENDOR_ID = 0x2886
XVF_PRODUCT_ID = 0x0022
CONTROL_IN_VENDOR_DEVICE = 0xC0
CONTROL_SUCCESS = 0
CONTROL_RETRY = 64
CONTROL_TIMEOUT_MS = 100_000


class XvfError(RuntimeError):
    """XVF transport or payload failure with a machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class XvfDoa:
    doa_deg: int
    speech_detected: bool


class Xvf3800:
    """Narrow, read-only XVF3800 control surface.

    The adapter intentionally exposes no generic write, reset, or firmware API.
    USB imports remain lazy so pure device policy works without PyUSB installed.
    """

    def __init__(
        self,
        device: Any,
        *,
        owns_device: bool = False,
        disposer: Callable[[Any], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 100,
    ) -> None:
        if (
            getattr(device, "idVendor", None) != XVF_VENDOR_ID
            or getattr(device, "idProduct", None) != XVF_PRODUCT_ID
        ):
            raise XvfError("wrong_device", "expected USB device 2886:0022")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise XvfError("invalid_config", "max_attempts must be a positive integer")
        if not callable(sleep):
            raise XvfError("invalid_config", "sleep must be callable")
        self.device = device
        self.owns_device = owns_device
        self._disposer = disposer
        self._sleep = sleep
        self.max_attempts = max_attempts
        self._closed = False

    @classmethod
    def discover(
        cls,
        *,
        bus: int | None = None,
        address: int | None = None,
        finder: Callable[..., Any] | None = None,
        disposer: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Xvf3800:
        """Find one exact commissioned Flex, optionally by USB bus/address."""
        if finder is None:
            try:
                import usb.core
            except ImportError as exc:
                raise XvfError("dependency_missing", "PyUSB is required for discovery") from exc
            finder = usb.core.find
        try:
            found = finder(
                find_all=True,
                idVendor=XVF_VENDOR_ID,
                idProduct=XVF_PRODUCT_ID,
            )
            devices = list(found or [])
        except Exception as exc:
            raise XvfError("usb_io", f"USB discovery failed: {exc}") from exc
        if bus is not None:
            devices = [device for device in devices if getattr(device, "bus", None) == bus]
        if address is not None:
            devices = [
                device for device in devices if getattr(device, "address", None) == address
            ]
        if not devices:
            raise XvfError("not_found", "reSpeaker Flex 2886:0022 was not found")
        if len(devices) > 1:
            raise XvfError(
                "multiple_devices",
                "multiple reSpeaker Flex devices found; select USB bus and address",
            )
        return cls(
            devices[0],
            owns_device=True,
            disposer=disposer,
            **kwargs,
        )

    def _read(self, *, command: int, resource: int, length: int) -> bytes:
        if self._closed:
            raise XvfError("closed", "XVF3800 adapter is closed")
        for attempt in range(self.max_attempts):
            try:
                response = bytes(self.device.ctrl_transfer(
                    CONTROL_IN_VENDOR_DEVICE,
                    0,
                    0x80 | command,
                    resource,
                    length,
                    CONTROL_TIMEOUT_MS,
                ))
            except Exception as exc:
                raise XvfError("usb_io", f"USB control read failed: {exc}") from exc
            if len(response) != length:
                raise XvfError(
                    "invalid_response",
                    f"expected {length} response bytes, received {len(response)}",
                )
            status = response[0]
            if status == CONTROL_SUCCESS:
                return response[1:]
            if status != CONTROL_RETRY:
                raise XvfError("device_status", f"XVF3800 returned status {status}")
            if attempt + 1 < self.max_attempts:
                self._sleep(0.01)
        raise XvfError(
            "retry_exhausted",
            f"XVF3800 remained busy for {self.max_attempts} attempts",
        )

    def read_version(self) -> tuple[int, int, int]:
        payload = self._read(command=0, resource=48, length=4)
        return payload[0], payload[1], payload[2]

    def read_doa(self) -> XvfDoa:
        payload = self._read(command=18, resource=20, length=5)
        doa_deg, speech = struct.unpack("<HH", payload)
        if not 0 <= doa_deg <= 359 or speech not in {0, 1}:
            raise XvfError(
                "invalid_response",
                f"invalid DOA payload angle={doa_deg} speech={speech}",
            )
        return XvfDoa(doa_deg=doa_deg, speech_detected=bool(speech))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.owns_device:
            return
        disposer = self._disposer
        if disposer is None:
            try:
                import usb.util
            except ImportError as exc:
                raise XvfError("dependency_missing", "PyUSB is required to dispose USB resources") from exc
            disposer = usb.util.dispose_resources
        try:
            disposer(self.device)
        except Exception as exc:
            raise XvfError("usb_io", f"USB resource disposal failed: {exc}") from exc
