import ray
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
ray.init()
policies = {
    "policy_0": PolicySpec(config={"gamma": 0.99}),
    "policy_1": PolicySpec(config={"gamma": 0.95}),
}
policy_ids = list(policies.keys())


def policy_mapping_fn(agent_id, episode, **kwargs):
    pol_id = policy_ids[agent_id]
    return pol_id


trainer = A3CTrainer(env=MultiAgentCartPole, config={
    "framework": "tfe",
    "multiagent": {"policies": policies,  "policy_mapping_fn": policy_mapping_fn}})
trainer.train()

