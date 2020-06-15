#%%
import torch as T
from torch import nn
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
#%%
class Agent():
   def __init__(self, env):
      super().__init__()
      self.env = env
      self.reward_mem = []
      self.prob_mem = []
      self.done_mem = []
      self.state_mem = []
      self.action_mem = []
      self.n_inputs = self.env.observation_space.shape[0]
      self.n_outputs = self.env.action_space.n
      self.vnet = nn.Sequential(
         nn.Linear(self.n_inputs,64),
         nn.LeakyReLU(0.1),
         nn.Linear(64,32),
         nn.LeakyReLU(0.1),
         nn.Linear(32,1))
      self.anet = nn.Sequential(
         nn.Linear(self.n_inputs, 64), 
         nn.LeakyReLU(0.1), 
         nn.Linear(64, 32),
         nn.LeakyReLU(0.1), 
         nn.Linear(32, self.n_outputs),
         nn.Softmax(dim=-1))
      self.old_net = None
      self.criterion = nn.MSELoss()
      self.aopti = T.optim.Adam(self.anet.parameters(), lr=3e-4)
      self.vopti = T.optim.Adam(self.vnet.parameters(), lr=1e-2)

   def clean_mem(self):
      self.reward_mem = []
      self.done_mem = []
      self.prob_mem = []
      self.reward_mem = []
      self.action_mem = []
      self.state_mem = []

   def discount(self, path, dones, gamma, norm=True):
      # Discount Rewards
      G = T.zeros(len(path))
      for i in range(len(path)):
         discount = 1
         g_sum = 0
         for j in range(i, len(path)):
            g_sum += path[j] * discount
            discount *= gamma
            if dones[j]:
               break
         G[i] = g_sum
      if norm:
         G = (G-G.mean())/G.std()
      return G

   def REINFORCE(self, gamma = 0.99, baseline=True):
      self.aopti.zero_grad()
      rewards, dones = self.reward_mem, self.done_mem

      # Discount Rewards
      G = self.discount(rewards, dones, gamma, norm=baseline)
      #print(G)

      # Calculate Loss: logProb*G
      log_probs = T.log(T.stack(self.prob_mem))
      #print("logprobs:", log_probs[:10])
      loss = 0
      for i in range(len(log_probs)):
         loss += -G[i] * log_probs[i]
      loss.backward()
      #print("loss:", loss)
      #before = list(self.anet.parameters())[1].clone()
      self.aopti.step()
      #diff = before - list(self.anet.parameters())[1]
      #print(diff[:10].abs().sum())

      # Clean Memory
      self.clean_mem()
      return loss.item()/32
   
   def PPO(self, gamma = 0.98, epsilon = 0.2, lambd = 0.97):
      if self.old_net == None:
         self.old_net = copy.deepcopy(self.anet)
      
      rewards, dones = T.tensor(self.reward_mem), T.tensor(self.done_mem)
      #print("rewards and dones:", rewards[dones])

      G = self.discount(rewards, dones, gamma, norm=True)
      #G = T.tensor(rewards)
      #print("discounted rewards:", G[:8])
      # Calculating Estimate and Vloss
      states = T.tensor(self.state_mem).float()
      V = self.vnet(states).view(-1)
      #print(G-V)
      #print("Reward Estimate:", E[:8])
      vloss = self.criterion(V, G)
      self.vopti.zero_grad()
      #print("vloss", vloss.item())
      vloss.backward()
      #before = list(self.vnet.parameters())[1].clone()
      self.vopti.step()
      #diff = before - list(self.vnet.parameters())[1]
      #print("V diff", diff.abs().sum().item())
      V = V.detach()

      # Calculating Policy Loss
      #print(self.prob_mem)
      probs = T.stack(self.prob_mem)
      #print(probs)
      old_probs = self.old_net(states)
      #print(old_probs)
      actions = T.tensor(self.action_mem)
      selected_old_probs = old_probs[T.arange(probs.shape[0]), actions]
      #print(selected_old_probs[:10])
      #print("Baseline Estimate:", E)
      R = (probs/selected_old_probs)[:-1]
      #print("Ratio:", R[:5].detach().numpy())
      #print(probs[:5], selected_old_probs[:5])
      #old_actions = old_probs.argmax(dim=-1).detach()
      #print(actions[:5])
      #E = G.mean()
      #print(old_actions[:5])
      V0 = V[:-1]
      V1 = V[1:]*(~dones[:-1])
      A = rewards[:-1] + (gamma *V1) - V0
      A = self.discount(A, dones, gamma*lambd, norm=True)
      surr1 = (R*A)
      surr2 = T.clamp(R, 1-epsilon, 1+epsilon)*A
      grad = T.min(surr1, surr2)
      #print("grad:", grad)
      grad = -1*grad.mean()

      self.aopti.zero_grad()
      # Save as old model:
      self.old_net = copy.deepcopy(self.anet)
      grad.backward()
      #before = list(self.anet.parameters())[1].clone()
      self.aopti.step()
      #diff = before - list(self.anet.parameters())[1]
      #print("A diff", diff.abs().sum().item())

      # Clean Memory
      self.clean_mem()
      return vloss.item()

   def act(self, visual=False):
      env = self.env
      state = env.reset()
      done = False
      score = 0
      while not done:
         if visual:
            env.render()
         probs = self.anet(T.tensor(state).float())
         #print(probs)
         action_distr = T.distributions.Categorical(probs)
         action = action_distr.sample()
         #print("atcion:", action, action.item())
         self.action_mem.append(action.item())
         self.prob_mem.append(probs[action])
         state, reward, done, info = env.step(action.item())
         self.reward_mem.append(reward)
         self.done_mem.append(done)
         self.state_mem.append(state)
         score += reward
      if visual:
         env.close()
      return score
      
#%%
# TRAIN LOOP
agent = Agent(gym.make('LunarLander-v2'))
#%%
score = []
vloss = []
visual = False
tmp = []
agent.clean_mem()
batchsize= 32
for k in range(0, 200*32):   
   tmp.append(agent.act(visual=visual))

   if len(tmp)==batchsize:
      #vl = agent.REINFORCE(baseline=True)
      vl = agent.PPO()
      vloss.append(vl)
      score.append(sum(tmp)/batchsize)
      #print(score[-1])
      tmp = []
   if k%200==0 and (k!=0):
      clear_output(wait=True)
      plt.clf()
      plt.plot(score, label='score')
      plt.legend()
      plt.savefig("ppo_lunar_score.png")
      #plt.show(block=False)
      plt.clf()
      plt.plot(vloss, label='vloss')
      plt.legend()
      plt.savefig("ppo_lunar_vloss.png")
      #plt.show(block=False)
      visual = True
   else:
      visual = False
#%%
# SHOW LOOP
for i in range(5):
   agent.act(visual=True)
# %%
import pickle
with open('lunar_ppo_agent.pkl', 'wb') as fp:
   obj = agent
   pickle.dump(obj, fp, pickle.HIGHEST_PROTOCOL)

# %%
