lerobot-record \
  --robot.type=bi_omx_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=bimanual_omx_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 25}}' \
  --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": /dev/video2, "width": 640, "height": 480, "fps": 25}}' \
  --teleop.type=bi_omx_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --teleop.id=bimanual_omx_leader \
  --display_data=false \
  --dataset.vcodec=h264 \
  --dataset.num_episodes=1 \
  --dataset.push_to_hub=False \
  --dataset.single_task="Grab and handover the red cube to the other arm" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=1 \
  --dataset.repo_id=gamiphy/bimanual-omx-chicken-sauce-1 \
  --dataset.episode_time_s=45
