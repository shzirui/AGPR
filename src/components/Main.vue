<script setup>
const base = import.meta.env.BASE_URL
</script>

<template>
  <main class="main">
    <section class="section header">
      <h1 class="title">Adaptive-Group Process Rewards</h1>
      <div class="subtitle">for GUI Agents</div>

      <div class="author-list">
        <span class="author">
          Zirui Shang<sup>1,2</sup>, Yang Liu<sup>2</sup>, Xin Shu<sup>2,3</sup>,
          Zhi Gao<sup>2,4</sup>, Xinxiao Wu<sup>1,4</sup>, Lifeng Fan<sup>2</sup>
        </span>
      </div>
      <div class="affiliation-list">
        <div>1. Beijing Key Laboratory of Intelligent Information Technology, School of Computer Science &amp; Technology, Beijing Institute of Technology</div>
        <div>2. State Key Laboratory of General Artificial Intelligence, BIGAI</div>
        <div>3. Wuhan University</div>
        <div>4. Guangdong Laboratory of Machine Perception and Intelligent Computing, Shenzhen MSU-BIT University</div>
        <div class="corresponding">Corresponding authors: gaozhibit@bit.edu.cn, wuxinxiao@bit.edu.cn</div>
      </div>

      <div class="links">
        <a class="link-button" href="https://github.com/shzirui/AGPR/tree/main/paper" target="_blank">
          <i class="fas fa-file-pdf"></i>
          <span>Paper</span>
        </a>
        <a class="link-button" href="https://github.com/shzirui/AGPR" target="_blank">
          <i class="fab fa-github"></i>
          <span>Code</span>
        </a>
      </div>
    </section>

    <section class="tldr">
      <b>TL;DR</b>
      We partition an executed GUI trajectory into subgoal groups, score actions and groups
      with a two-level critic, and use those process rewards to shape GRPO's advantage.
    </section>

    <section class="section">
      <div class="image-frame image-frame--hero">
        <img class="hero-figure" :src="`${base}agpr/intro-1.png`" alt="Fixed-window rewards, milestone rewards, and adaptive-group process rewards">
      </div>
      <p class="caption">
        A fixed local window can include irrelevant context or miss later evidence. Predefined
        milestones can leave a valid executed path uncredited. Adaptive groups follow the
        subgoals that were actually executed.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Introduction</h2>
      <p class="intro">
        Training GUI agents with reinforcement learning suffers from sparse outcome rewards.
        A trajectory is successful or not only at the end, so every intermediate action receives
        the same learning signal. Process rewards can score those actions, but the value of a
        GUI action often depends on evidence beyond the immediate next screenshot. Fixed-window
        step rewards judge an action inside a preset local context. Milestone rewards organize
        credit around predefined checkpoints. Neither scope is determined by the trajectory that
        was actually executed. Adaptive-group process rewards recover subgoals from the
        instruction and the agent's responses, then judge each action inside its group and each
        group inside the full trajectory.
      </p>
    </section>

    <section class="section">
      <h2 class="section-title">Method</h2>
      <div class="image-frame image-frame--wide">
        <img class="wide-figure" :src="`${base}agpr/framework-1.png`" alt="Adaptive-group process reward framework">
      </div>
      <div class="method-grid">
        <div class="method-item">
          <div class="method-index">01</div>
          <h3>Adaptive Groups</h3>
          <p>
            A grouping critic reads the task instruction and the agent's responses, then splits
            the executed trajectory into contiguous subgoal groups. The groups are not required
            milestones of the instruction, so different valid executions can induce different
            partitions.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">02</div>
          <h3>Two-Level Scores</h3>
          <p>
            At the action level, the critic checks execution from the local visual change and
            contribution to the subgoal from later actions in the group. At the group level, it
            aggregates those judgments and then revises them with the context of every group in
            the trajectory. The process reward of an action is the product of its action score
            and the refined score of its group.
          </p>
        </div>
        <div class="method-item">
          <div class="method-index">03</div>
          <h3>Advantage Shaping</h3>
          <p>
            GRPO's outcome advantage stays constant across the steps of a trajectory. We add a
            process advantage standardized over all steps in the sampled group. The training
            entry uses this mixture with coefficient 0.5.
          </p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">Experiments</h2>
      <div class="experiment-stack">
        <div class="experiment-block">
          <h3 class="experiment-title">Process Reward Benchmarks</h3>
          <p class="intro">
            On CUARewardBench the framework reaches 81.3 overall accuracy with Qwen3.6-27B.
            That is 3.6 points above GPT-5 under the OpenCUA-Reflector template and 14.0 points
            above the same Qwen3.6-27B backbone under that template. On CUAStepBench it reaches
            39.9 F1, above SeekJudge with the same backbone (36.3).
          </p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Reward model</th>
                  <th>P</th>
                  <th>NPV</th>
                  <th>R</th>
                  <th>S</th>
                  <th>OA</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>GPT-5, OpenCUA-Reflector</td>
                  <td>72.7</td>
                  <td>87.8</td>
                  <td>92.3</td>
                  <td>61.6</td>
                  <td>77.7</td>
                </tr>
                <tr>
                  <td>Qwen3.6-27B, OpenCUA-Reflector</td>
                  <td>62.5</td>
                  <td>87.0</td>
                  <td>95.1</td>
                  <td>36.6</td>
                  <td>67.3</td>
                </tr>
                <tr class="ours">
                  <td>Ours, Qwen3.6-27B</td>
                  <td>79.4</td>
                  <td>84.1</td>
                  <td>87.8</td>
                  <td>73.9</td>
                  <td>81.3</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="caption">CUARewardBench under the OpenCUA-Reflector comparison. Full tables are in the paper.</p>
        </div>

        <div class="experiment-block">
          <h3 class="experiment-title">Reinforcement Learning on OSWorld</h3>
          <p class="intro">
            UI-TARS-1.5-7B is trained on Chrome, LibreOffice Impress, and LibreOffice Calc.
            Adaptive-group process rewards improve overall success over the base model,
            outcome-only GRPO, and SeekJudge.
          </p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Chrome</th>
                  <th>Impress</th>
                  <th>Calc</th>
                  <th>Overall</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>UI-TARS</td>
                  <td>32.6 / 28.2</td>
                  <td>33.8 / 32.1</td>
                  <td>10.6 / 7.6</td>
                  <td>25.6 / 22.6</td>
                </tr>
                <tr>
                  <td>Outcome GRPO</td>
                  <td>43.5 / 37.0</td>
                  <td>44.7 / 41.6</td>
                  <td>19.1 / 12.8</td>
                  <td>35.7 / 30.4</td>
                </tr>
                <tr>
                  <td>SeekJudge</td>
                  <td>45.7 / 38.3</td>
                  <td>46.8 / 41.1</td>
                  <td>19.1 / 14.9</td>
                  <td>37.1 / 31.4</td>
                </tr>
                <tr class="ours">
                  <td>Ours</td>
                  <td>49.9 / 43.4</td>
                  <td>48.7 / 43.3</td>
                  <td>23.4 / 17.0</td>
                  <td>40.6 / 34.5</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="caption">Success rate (%) on OSWorld. Each cell is one run / mean of three runs.</p>
        </div>

        <div class="experiment-block">
          <h3 class="experiment-title">Qualitative Example</h3>
          <div class="image-frame image-frame--wide">
            <img class="wide-figure" :src="`${base}agpr/qual_chrome.png`" alt="Qualitative Chrome trajectory">
          </div>
          <p class="caption">
            A Chrome trajectory scored by the adaptive-group critic. Additional cases are in the paper appendix.
          </p>
        </div>
      </div>
    </section>

    <section class="footer">
      <p>Project page for Adaptive-Group Process Rewards for GUI Agents.</p>
    </section>
  </main>
</template>

<style scoped>
.main {
  width: min(1160px, calc(100% - 32px));
  margin: 0 auto;
  color: #202635;
}

.section {
  margin: 72px 0;
  text-align: center;
}

.header {
  margin-top: 58px;
}

.title {
  margin: 0;
  font-size: clamp(2.6rem, 6vw, 4.6rem);
  line-height: 0.98;
  font-weight: 800;
}

.subtitle {
  width: min(100%, 900px);
  margin: 20px auto 0;
  color: #4f5c72;
  font-size: clamp(1.55rem, 3.4vw, 2.55rem);
  line-height: 1.15;
}

.author-list {
  margin-top: 22px;
  color: #536078;
  font-size: 1.08rem;
}

.affiliation-list {
  width: min(100%, 920px);
  margin: 14px auto 0;
  color: #536078;
  font-size: 0.98rem;
  line-height: 1.55;
}

.corresponding {
  margin-top: 8px;
  color: #2d3d56;
  font-weight: 650;
}

.links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 12px;
  margin-top: 26px;
}

.link-button {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
  padding: 0 18px;
  border-radius: 999px;
  color: #ffffff;
  background: #222a37;
  text-decoration: none;
  font-weight: 650;
}

.link-button:hover {
  background: #3a4659;
}

.tldr {
  max-width: 980px;
  margin: 42px auto 10px;
  padding: 18px 22px;
  border-left: 4px solid #2663ff;
  background: #ffffff;
  box-shadow: 0 8px 28px rgba(31, 42, 68, 0.08);
  text-align: left;
  font-size: 1.1rem;
  line-height: 1.6;
}

.section-title {
  margin: 0 0 20px;
  font-size: clamp(1.8rem, 4vw, 2.4rem);
  line-height: 1.16;
}

.section-title::after {
  content: "";
  display: block;
  width: 44px;
  height: 3px;
  margin: 12px auto 0;
  border-radius: 3px;
  background: #2663ff;
}

.intro,
.caption {
  max-width: 980px;
  margin: 14px auto;
  color: #334057;
  font-size: 1.04rem;
  line-height: 1.75;
}

.caption {
  color: #59657a;
  font-size: 0.96rem;
}

.image-frame {
  padding: 18px;
  border: 1px solid #dfe5f0;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 26px rgba(31, 42, 68, 0.08);
}

.image-frame--hero,
.image-frame--wide {
  width: 100%;
  max-width: 1060px;
  margin: 0 auto;
}

.hero-figure,
.wide-figure {
  display: block;
  width: 100%;
}

.method-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 22px;
}

.method-item {
  min-height: 220px;
  padding: 22px;
  border: 1px solid #dfe5f0;
  border-radius: 8px;
  background: #ffffff;
  text-align: left;
  box-shadow: 0 6px 20px rgba(31, 42, 68, 0.06);
}

.method-index {
  width: 32px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #59657a;
  background: #f4f6fa;
  border: 1px solid #dfe5f0;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 700;
}

.method-item h3 {
  margin: 14px 0 10px;
  font-size: 1.22rem;
}

.method-item p {
  margin: 0;
  color: #48546a;
  line-height: 1.62;
}

.experiment-stack {
  max-width: 1060px;
  margin: 18px auto 0;
}

.experiment-block + .experiment-block {
  margin-top: 52px;
  padding-top: 34px;
  border-top: 1px solid #dfe5f0;
}

.experiment-title {
  margin: 0 0 12px;
  color: #1f2a44;
  font-size: 1.42rem;
}

.table-wrap {
  overflow-x: auto;
  margin: 18px auto 0;
  border: 1px solid #dfe5f0;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 26px rgba(31, 42, 68, 0.08);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.98rem;
}

th,
td {
  padding: 12px 14px;
  border-bottom: 1px solid #e6ebf3;
  text-align: center;
}

th {
  color: #536078;
  font-weight: 700;
  background: #f7f8fb;
}

td:first-child,
th:first-child {
  text-align: left;
}

tr.ours td {
  font-weight: 750;
  background: #f3f7ff;
}

.footer {
  margin: 90px 0 46px;
  color: #778197;
  text-align: center;
}

@media (max-width: 820px) {
  .title {
    font-size: clamp(2.2rem, 10vw, 3rem);
  }

  .method-grid {
    grid-template-columns: 1fr;
  }

  .intro {
    text-align: left;
  }
}
</style>
