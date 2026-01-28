# 🛸 ML-Based Trajectory Optimization for UAV-Enabled 5G Communication

**Student Name:** Raj Pohekar
**Degree:** B.E. Electronics & Telecommunication Engineering
**College:** Pune Institute of Computer Technology (PICT)

---

## 1. Project Overview

**Title:**
ML-Based Trajectory Optimization for UAV-Enabled 5G Communication

**Domain:**
Wireless Communication + Machine Learning (Reinforcement Learning)

**Core Idea:**
This project focuses on optimizing the flight trajectory of a **UAV acting as an aerial base station** in a 5G network using **machine learning**, specifically **reinforcement learning**, to improve **data rate, coverage, and energy efficiency**.

---

## 2. Why This Project? (Problem Motivation)

In real-world scenarios such as:
* Natural disasters
* Large public events (concerts, stadiums)
* Emergency response situations

Terrestrial 5G infrastructure:
* Gets congested
* Gets damaged
* Cannot be deployed quickly

### Why UAVs?
UAVs can:
* Be deployed rapidly
* Provide Line-of-Sight (LoS) communication
* Act as temporary 5G base stations

### Core Challenge
Even though UAVs are flexible, **trajectory planning is critical**.

A bad trajectory results in:
* High path loss
* Low SNR
* Poor data rates
* Excessive battery drain

Traditional trajectory planning:
* Is static
* Uses heuristic or mathematical optimization
* Cannot adapt to dynamic users or channel conditions

---

## 3. Key Problem Statement

> **How can a UAV dynamically adjust its flight path to maximize 5G communication performance while minimizing energy consumption, under time-varying channel conditions and battery constraints?**

---

## 4. Why Machine Learning?

Traditional optimization techniques:
* Require complete environment knowledge
* Are computationally expensive
* Do not scale well in dynamic scenarios

### Advantage of Machine Learning
Machine Learning allows the UAV to:
* Learn from interaction with the environment
* Adapt its trajectory dynamically
* Balance conflicting objectives (rate vs energy)

This makes **Reinforcement Learning (RL)** a natural fit.

---

## 5. System Model Explanation

### 5.1 Network Scenario
* A **single UAV** serves multiple ground users.
* Users are randomly distributed in a 2D region.
* UAV operates at a controllable altitude.
* Communication is over a shared 5G channel.

### 5.2 Communication Characteristics
* Air-to-ground links can be:
    * Line-of-Sight (LoS)
    * Non-Line-of-Sight (NLoS)
* Channel quality depends on:
    * UAV altitude
    * Horizontal distance
    * Elevation angle ($\theta$)

### 5.3 Energy Constraints
The UAV has limited onboard battery:
* **Propulsion energy:** Movement.
* **Communication energy:** Transmission + circuitry.

Hence, trajectory optimization must be **energy-aware**.

---

## 6. Machine Learning Model Used

### 6.1 Why Reinforcement Learning?
Trajectory optimization is a **sequential decision-making problem**:
* Decisions at current time affect future states.
* No closed-form optimal solution exists.

Hence, RL is suitable.

---

## 7. Reinforcement Learning Formulation

### 7.1 Agent and Environment
* **Agent:** UAV
* **Environment:** Wireless channel + user distribution + energy model

### 7.2 RL Components

#### State ($S$)
* UAV position
* Current SNR / data rate
* Remaining battery energy

#### Action ($A$)
* UAV movement decisions:
    * Forward
    * Backward
    * Left
    * Right
    * Hover

#### Reward ($R$)
The reward function is designed as:

$$Reward = \alpha \times \text{Achievable Data Rate} - \beta \times \text{Energy Consumption}$$

This ensures:
* High throughput is rewarded.
* Excessive energy usage is penalized.

### 7.3 Learning Algorithm
We used **Q-learning** with:
* $\epsilon$-greedy exploration
* Learning rate ($\eta$)
* Discount factor ($\gamma$)

The UAV gradually learns an **optimal trajectory policy**.

---

## 8. Channel Modeling

### 8.1 LoS / NLoS Probability
The probability of LoS depends on the elevation angle between UAV and user.

$$P(LoS) = \frac{1}{1 + a \cdot \exp(-b[\theta - a])}$$

### 8.2 Path Loss Model
Average path loss is calculated by combining:
* LoS path loss
* NLoS path loss
* Their respective probabilities

$$PL_{avg} = P(LoS) \cdot PL_{LoS} + [1 - P(LoS)] \cdot PL_{NLoS}$$

---

## 9. Communication Performance Metrics

### 9.1 Signal-to-Noise Ratio (SNR)
SNR depends on:
* Transmit power
* Path loss
* Noise power

### 9.2 Achievable Data Rate
Computed using **Shannon’s Capacity Formula**:

$$R = B \times \log_2(1 + \text{SNR})$$

---

## 10. Energy Consumption Model

### 10.1 Propulsion Energy
Energy consumed due to UAV movement between time slots.

### 10.2 Communication Energy
Includes:
* Adaptive transmit power
* Constant circuit power

### 10.3 Total Energy
$$E_{total} = E_{propulsion} + E_{communication}$$

---

## 11. Performance Evaluation Parameters

The system is evaluated using:
1.  UAV trajectory length
2.  Average achievable data rate
3.  Signal-to-noise ratio
4.  Coverage probability
5.  Total energy consumption
6.  Trajectory smoothness
7.  RL convergence behavior

---

## 12. Results and Analysis

### 12.1 Trajectory Behavior
* RL-based UAV adapts its path toward dense user regions.
* Avoids unnecessary movement.
* Maintains better LoS probability.

### 12.2 Data Rate Improvement
* RL-based approach achieves **18–25% higher average data rate**.
* Static trajectories suffer from inefficient positioning.

### 12.3 Energy Efficiency
* Energy consumption reduced by **20–30%**.
* UAV avoids energy-wasting maneuvers.

### 12.4 Learning Convergence
* Initial exploration causes fluctuations.
* Gradual convergence to a stable optimal policy.

---

## 13. Key Contributions of the Project

* ML-driven adaptive UAV trajectory planning.
* Joint consideration of communication and energy.
* Practical applicability to emergency 5G deployments.
* Lightweight RL framework with stable convergence.

---

## 14. Limitations

* Single UAV scenario.
* Simulation-based validation.
* No real-world flight testing.

---

## 15. Future Scope

* Multi-UAV coordination using multi-agent RL.
* Deep Reinforcement Learning (DQN / DDQN).
* Integration with 6G, mmWave, IRS.
* Federated learning for distributed intelligence.
* Real UAV hardware deployment.

---

## 16. Final Conclusion

This project demonstrates that **reinforcement learning enables intelligent, adaptive, and energy-efficient UAV trajectory optimization** for UAV-enabled 5G communication systems, outperforming traditional static approaches and making it suitable for real-world emergency and temporary network scenarios.

---
---

# Interview Clarification: Machine Learning Background & Preparedness

## 17. Machine Learning Background (Interview Clarification Section)

Although **Machine Learning was not a prior specialization listed on my resume**, I intentionally selected this project to **gain hands-on exposure to ML concepts through practical application**.

This project primarily focuses on **Reinforcement Learning (RL)**, which is a subfield of machine learning well-suited for **sequential decision-making problems** such as UAV trajectory optimization.

Rather than attempting to cover all machine learning techniques, the scope of ML in this project is **deliberately focused and well-defined**.

---

## 18. Scope of Machine Learning Used in This Project

The machine learning scope of this project is limited to:
* Reinforcement Learning fundamentals
* Q-learning algorithm
* State–Action–Reward modeling
* Policy learning through interaction with the environment

The project **does not involve**:
* Deep Learning (CNN, RNN, Transformers)
* Backpropagation or gradient descent
* Large-scale neural networks

This controlled scope ensures **clarity, correctness, and practical relevance**.

---

## 19. Why Reinforcement Learning Was Chosen

Reinforcement learning was selected because:
* There is **no labeled dataset** for the “optimal UAV trajectory”.
* The UAV must learn through **trial and error**.
* Each movement decision affects future system states.
* The problem is naturally **sequential and dynamic**.

This makes RL more suitable than supervised or unsupervised learning approaches.

---

## 20. Reinforcement Learning Formulation (Interview-Ready)

* **Agent:** UAV
* **Environment:** Wireless channel, user distribution, energy constraints
* **State:** UAV position, SNR, remaining energy
* **Action:** UAV movement decisions
* **Reward:** Balance between data rate and energy consumption

**Reward function:**
$$Reward = \alpha \times \text{Achievable Data Rate} - \beta \times \text{Energy Consumption}$$

This reward design ensures that the UAV does not maximize throughput at the cost of excessive battery usage.

---

## 21. Honest Positioning of ML Expertise (Interview Strategy)

This project reflects an **active learning approach to machine learning**, not claimed mastery.

**Interview positioning:**

> *Machine learning is relatively new to me, and this project was a deliberate step to apply ML concepts practically. I focused on reinforcement learning concepts required for this system rather than attempting broad ML coverage.*

This approach highlights:
* Learning mindset
* Technical honesty
* Practical implementation skills

---

## 22. Prepared Responses for ML Cross-Questions

### Q: What type of machine learning is used?
**Answer:** Reinforcement Learning.

### Q: Why not supervised learning?
**Answer:** There is no labeled optimal trajectory; the UAV must learn from interaction.

### Q: Did you use deep learning?
**Answer:** No. Classical Q-learning was sufficient for this scenario. Deep RL is part of future scope.

### Q: Why ML in the title if ML is new to you?
**Answer:** Reinforcement learning is a core branch of machine learning, and trajectory optimization here is learned, not rule-based.

---

## 23. Key Takeaway for Interviewers

This project demonstrates:
* Practical application of ML concepts
* Clear understanding of reinforcement learning
* Correct problem–algorithm mapping
* Strong foundation for further ML growth

The focus is on **correct usage**, not superficial buzzwords.

---
---

# 1-Minute Project Introduction Script (Interview-Ready)

> “My final year project is titled *ML-Based Trajectory Optimization for UAV-Enabled 5G Communication*.
>
> The goal of this project is to optimize the flight path of a UAV that acts as a temporary 5G base station during emergency or high-density scenarios.
>
> Instead of using static or rule-based trajectories, we use reinforcement learning so the UAV can learn an optimal path by interacting with the environment.
>
> The UAV observes its position, communication quality, and remaining energy, takes movement decisions, and receives rewards based on data rate and energy consumption.
>
> Over time, it learns an energy-efficient trajectory that improves coverage and throughput.
>
> This project helped me practically apply reinforcement learning concepts such as state, action, reward, and Q-learning in a real communication problem.”

---

## How to Use This in Interviews

* Start with the **1-minute script**
* If ML background is questioned → go to Sections 17–21
* If technical ML questions come → stay within Sections 18–22
* Never go outside the defined scope
