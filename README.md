# rl_project

Since argparse is part of the standard Python library, it should already be installed. However, if it’s not, you can install it using the follwing command:
"pip install argparse"

We have args you can run togther with the python command and by this choosing what the value of the hyperparamters on other variables:
python main.py <args>

1. To change replay buffer size, run with the following flag:
python main.py --buffer <integer>
For example to run with 5000000 replay buffer size:
python main.py --buffer 5000000

2. To change size of the batch, run with the following flag:
python main.py --batch <integer>
For example to run with 32 batch size:
python main.py --batch 32

3. To change timestep when learning starts, run with the following flag:
python main.py --learning_starts <integer>
For example to run with 50000 timestep when learning starts:
python main.py --learning_starts 50000

4. To change updating weights frequency, run with the following flag:
python main.py --learning_freq <integer>
For example to run with 4 update frequency:
python main.py --learning_freq 4

5. To change length of history used, run with the following flag:
python main.py --history_len <integer>
For example to run with 5000 history length:
python main.py --history_len 5000

6. To change update target frequency, run with the following flag:
python main.py --update_freq <integer>
For example to run with 4 update frequency:
python main.py --update_freq 4

7. To change netork build, run with the following flag:
python main.py --func <string>
Choose between the following values:
a. To run the normal dqn network =>   python main.py --func dqn
b. To run the double dqn network =>   python main.py --func ddqn
c. To run the dueling dqn network =>   python main.py --func duelingdqn

8. To change gamma size, run with the following flag:
python main.py --gamma <float>
For example to run with 0.9 gamma size:
python main.py --gamma 0.9

9. To change alpha size, run with the following flag:
python main.py --alpha <float>
For example to run with 0.9 alpha size:
python main.py --alpha 0.9

10. To change learning rate, run with the following flag:
python main.py --learning_rate <float>
For example to run with 0.002 learning rate:
python main.py --learning_rate 0.002

11. To change agent, run with the following flag:
python main.py --agent <string>
Choose between the following values:
a. To run the greedy agent =>   python main.py --agent greedy
b. To run the noisy agent =>   python main.py --agent noisy
c. To run the softmax agent =>   python main.py --agent softmax

12. To change beta value for softmax agent, run with the following flag:
python main.py --beta <float>
For example to run with 0.002 beta:
python main.py --beta 0.002

13. To change standart deviation value for noisy agent, run with the following flag:
python main.py --std <float>
For example to run with 0.002 std:
python main.py --std 0.002

14. To change epsilon, run with the following flag:
python main.py --eps <float>
For example to run with 0.002 eps:
python main.py --eps 0.002


Notice you can run multiply arguments for example for runing wint double dqn network and softmax agent with beta 9 we can run the following command:
python main.py --func ddqn --softmax --beta 9


