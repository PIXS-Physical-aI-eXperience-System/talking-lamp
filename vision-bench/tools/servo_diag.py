"""Read-only servo diagnostic: per-register comm status and the error flags."""
from scservo_sdk import PortHandler, PacketHandler

FLAGS = [(0x01, "전압"), (0x02, "센서"), (0x04, "과열"),
         (0x08, "전류"), (0x10, "각도"), (0x20, "과부하")]

ph = PortHandler("/dev/ttyACM0")
if not ph.openPort():
    print("포트 열기 실패")
    raise SystemExit
ph.setBaudRate(1000000)
pk = PacketHandler(0)

for sid in range(1, 6):
    te, comm, err = pk.read1ByteTxRx(ph, sid, 40)
    if comm != 0:
        print(f"{sid}: 응답 없음 ({pk.getTxRxResult(comm)})")
        continue
    tmp, c_t, e_t = pk.read1ByteTxRx(ph, sid, 63)
    ld, c_l, e_l = pk.read2ByteTxRx(ph, sid, 60)
    volt, c_v, _ = pk.read1ByteTxRx(ph, sid, 62)
    bits = err | e_t | e_l
    named = [n for m, n in FLAGS if bits & m] or ["없음"]
    ok = "ok" if (c_t == 0 and c_l == 0) else "읽기실패"
    print(f"{sid}: 토크 {'켜짐' if te else '꺼짐'}  온도 {tmp}C  전압 {volt/10:.1f}V  "
          f"부하 {(ld & 0x3FF)/10:.1f}%  통신 {ok}  오류플래그 {','.join(named)} (0x{bits:02x})")
ph.closePort()
