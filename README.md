Brain Corp robotic motion planning environment
==============================================

This is a python package that provides a robotic planning environment
with an interface that is similar to OpenAI gym.

The task is navigation of 2d large robots in tight spaces.

```python3
env = RandomMiniEnv()
obs = env.reset()


while not done:
    action = planner.plan(obs)
    observation, reward, done, info = env.step(action)
```


Statefullness of the env
------------------------
You can save and restore the full state of the environment.
This is useful for example for Monte Carlo simulation, where
you need to run many rollouts from one state.

The syntax works as follows
```python3
state = env.get_state()

# do sth with the env, try out some plan
while sth():
    env.act(some_action)

# restore the env to the previous state
env.set_state(state)
```

Pickling, Saving, Loading
-------------------------
All pieces of the framework can be rendered to basic python types
(`int`, `float`, `dict`, the most complicated `numpy.ndarray`).
What is more, the objects can be constructed back from this representation
in a completely idempotent way.

This way you can use pickle to save / load whatever you want.


Tour of the main classes
------------------------

- `Observation` from `envs.base.obs`
   represents the type of observation that is returned from the 
   environment.
- `Action` from `envs.base.action`
  represents the action that is passed for execution to the environment
- `PlanEnv` from `envs.base.env` is
the class that represents the environment itself.
- `State` from `envs.base.env` is
the class that represents the state of the environment
- `ContinuousRewardProvider` from `envs.base.reward`
is a class that interprets what robot has done in the environment
and assigns rewars for it


Runners
-------

Please see the scripts in `scripts` for examples how to run the environment.


Types of environments
----------------------
The base class is `envs.base.env.PlanEnv`
You need to supply a path to follow and costmap that represents 
obstacles.

An example of this is given in the script
`scripts.env_runners.rw_randomized_corridor_3_boxes`,
where we load a custom costmap and path based on percepts from a real
robot.

There are additional classes that supply these path and costmaps in 
special ways:
- `envs.mini_env.RandomMiniEnv` - randomized,
 synthetic 'parallel parking' small square environment with one 
 obstacle, where you have to reach next pose that can be close to you, 
 but can have an ankward path to it.
- `envs.synth_turn_env.AisleTurnEnv` - 
a randomized synthetic environment where you have to follow a path 
turnning into an aisle.


Relation to OpenAI gym
----------------------
A frequently asked question we get is
> why not just subclass `gym.Env`?
This is because `gym` depends on `scipy`.
At Brain Corp, we choose not to use `scipy`.

As far as we can see from the code, OpenAI `gym` is on the road
to remove this dependency. Hopefully then we will subclass
it fully.
