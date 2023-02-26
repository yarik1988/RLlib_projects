if exist results\DogFight\ (
  C:\PROGRAMS\Python39\Scripts\mlagents-learn DogFight.yaml --num-areas=4 --num-envs=4 --env=../build/DogFight/2DTest --run-id=DogFight --resume --no-graphics
) else (
  C:\PROGRAMS\Python39\Scripts\mlagents-learn DogFight.yaml --num-areas=4 --num-envs=4 --env=../build/DogFight/2DTest --run-id=DogFight --no-graphics
)


