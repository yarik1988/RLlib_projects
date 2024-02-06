if exist results\Valley\ (
  mlagents-learn Valley.yaml --run-id=Valley --resume --no-graphics
) else (
  mlagents-learn Valley.yaml --run-id=Valley --no-graphics
)


