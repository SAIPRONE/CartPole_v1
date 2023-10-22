# CartPole_v1
# DQN Agent for OpenAI Gym Environments 🚀

A Deep Q-Network (DQN) implementation designed  to tackle the classic reinforcement learning problems `CartPole-v1` and `MountainCar-v1` from OpenAI's Gym.

## 🎮 Environments

1. [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)
2. [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) (Commented out in the code, just uncomment to use)

## 🕹️ How to Use

1. **Dependencies**: First things first, ensure you have the following Python libraries installed:
   - [gym](https://gym.openai.com/docs/)
   - [numpy](https://numpy.org/)
   - [tensorflow](https://www.tensorflow.org/)

2. **Run the Script**: Launch the script to either train or test the DQN agent on your desired environment. Adjust the `TRAIN`, `LOAD`, and `SAVE` flags in the code as per your requirements.

3. **Visualize**: If you're not training (`TRAIN = False`), you can watch the agent interact with the environment in real-time!

## 🧠 Model Architecture

The agent's brain is a simple neural network built using TensorFlow's Keras:
- Input Layer: Matches the state size of the environment.
- Hidden Layer 1: 32 neurons with LeakyReLU activation.
- Hidden Layer 2: 32 neurons with LeakyReLU activation.
- Output Layer: Matches the number of possible actions with a linear activation.

The network is compiled with the `Adam` optimizer and a mean squared error (MSE) loss function.

## 🔧 Hyperparameters

- `gamma` (Discount Rate): 0.99
- `epsilon` (Exploration Rate): Starts at 1.0 for training, 0.0 for testing.
- `epsilon_min`: 0.01
- `epsilon_decay`: 0.99
- `learning_rate`: 0.001

## 💾 Loading and Saving

The trained model weights can be saved to and loaded from an `.h5` file. Adjust the `LOAD` and `SAVE` flags in the code to control this functionality.

## 🔗 References

- [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [OpenAI Gym](https://gym.openai.com/)
- [TensorFlow & Keras](https://www.tensorflow.org/)

## 📜 License

This project is open source, under the BSD 3-Clause License.
