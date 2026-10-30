#!/usr/bin/env python3
"""Run one typed cognition turn and optionally send its motion tags to ROS."""

import argparse
import json
import time

from cognition_run import respond


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="직접 입력할 사용자 문장")
    parser.add_argument("--labels", default="", help="카메라에서 확인한 라벨을 쉼표로 구분")
    parser.add_argument("--execute", action="store_true", help="모터 행동을 실제로 실행")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    response, raw = respond(labels, args.text)
    print(f"대사: {response.say}")
    print(f"행동: {response.actions}")
    if response.notes:
        print(f"파서: {response.notes}")
    if not response.ok:
        raise SystemExit("모델 응답을 해석할 수 없어 행동을 보내지 않았습니다")
    if not args.execute:
        print("모터 실행은 생략했습니다. 실제 실행에는 --execute를 붙이세요.")
        return

    import rclpy
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("lamp_cognition_text_turn")
    try:
        publisher = node.create_publisher(String, "/lamp/cognition/response", 10)
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if publisher.get_subscription_count() == 0:
            raise SystemExit("인지 행동을 받는 노드가 없어 모터 명령을 보내지 않았습니다")
        publisher.publish(String(data=json.dumps(response.to_dict(), ensure_ascii=False)))
        rclpy.spin_once(node, timeout_sec=0.2)
        print("인지 행동을 모터 연결 노드에 보냈습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
