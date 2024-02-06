if exist results\DogFight\ (
  mlagents-learn DogFight.yaml --num-areas=4 --run-id=DogFight --resume --no-graphics
) else (
  mlagents-learn DogFight.yaml --num-areas=4 --run-id=DogFight --no-graphics
)


