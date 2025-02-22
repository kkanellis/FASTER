#pragma once

class AgentManager {
 public:
  AgentManager(uint64_t memory_budget)
    : memory_budget_{ memory_budget } {

  }

 private:
  uint64_t memory_budget_;
};
