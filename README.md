# Evolution Strategies OpenAI

Implementation is strictly for educational purposes, not distributed and not very effitient (yet), but it works.


## Experiments

### CartPole

<p float="left">
  <img src="plots/test_CartPole_v1.png" width="360" />
  <img src="plots/gifs/best_pole.gif" width="360" /> 
</p>

### LunarLander

<p float="left">
  <img src="plots/test_LunarLander_v3.png" width="360" />
  <img src="plots/gifs/best_lunar.gif" width="360" /> 
</p>

### LunarLanderContinuous

<p float="left">
  <img src="plots/test_LunarLanderCont_v1.png" width="360" />
  <img src="plots/gifs/best_lunar_cont.gif" width="360" /> 
</p>


## TODO

- parallelization (!!!)

- weight decay (noise std decay???)
- rank transformation 
- mirrored sampling
- adam
- batch norm
- skip frames

## TODO ENVS

- [x] CartPole   
- [ ] CarRacing      
- [x] LunarLander   
- [ ] LunarLanderContinuous

Cannot solve, sparse rewards + random
- [ ] MountainCar (hard)     
- [ ] Taxi & FrozenLake (hard)

<!-- ## Ideas

skip frames - https://notanymike.github.io/Solving-CarRacing/, https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/, https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py (NB!, class MaxAndSkipEnv(gym.Wrapper) and class WarpFrame(gym.ObservationWrapper)), https://alexandervandekleut.github.io/gym-wrappers/
noise/lr annealing - https://cs231n.github.io/neural-networks-3/#anneal,  https://arxiv.org/pdf/1608.03983.pdf, https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1


 
- добавить инициализацию модели с уже обученных весов и попробовать решить несколько сред сразу
- for parallel (exampler interface): es.eval_population(npop) -> updates -> es.update_population(updates) -->
<!-- функция берет модель, рандомно меняет веса, прогоняет, получает ревард и возвращает апдйет весов сразу -> легче параллелить,
 чем если отдельно генерировать сначала популяцию, потом ее отдельно прогонять, а потом уже апдейтить -->

<!-- Наблюдения: легко решает среды, в которых легко исследовать/пробовать разное, т.к тогда точнее получается градиент и больше данных по реварду. Среды в которых ревард очень редкий решаются очень плохо т.к. до того, как случайно случится событие с ревардом может пройти очень много времени, т.к. поиск до этого случайны и обучения нет. Taxi-v3: плохо работает и генетический и метод кросс энтропии -->

## References

[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever)
