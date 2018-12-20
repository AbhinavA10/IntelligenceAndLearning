class Population { // class for a collection of dots
  Dot[] dots;
  float fitnessSum;
  int gen = 1;
  int bestDot =0; // don't want to mutate negatively
  int minStep = 9999999; // we want to save the minimum number of steps to get to the goal, and not allow any future dots to take more steps

  // ===================================== POPULATION CONSTRUCTOR =======================================================
  Population(int size) {
    dots = new Dot[size];
    for (int i =0; i<size; i++) {
      dots[i] = new Dot();
    }
  }
  // ===================================== SHOW =======================================================
  void show() {
    for (int i =1; i<dots.length-1; i++) {
      dots[i].show();
    }
    dots[0].show();
  }
  // ===================================== UPDATE =======================================================
  void update() {
    for (int i=0; i<dots.length; i++) {
      if (dots[i].brain.step>minStep) { // if the current dot has taken more steps than a dot that has already got to the goal, kill it
        dots[i].dead=true;
      } else {
        dots[i].update();
      }
    }
  }
  // ===================================== CALCULATE FITNESS =======================================================
  void calculateFitness() {
    for (Dot d : dots) { // (ObjectClassNameType i: objecArrayListName)
      d.calculateFitness();
    }
  }
  // ===================================== ALL DOTS DEAD =======================================================
  boolean allDotsDead() { // need to know when all the dots are dead, so we can stop it
    for (Dot d : dots) {
      if (!d.dead && !d.reachedGoal) {
        return false;
      }
    }
    return true;
  }
  // ===================================== NATURAL SELECTION =======================================================
  void naturalSelection() {
    Dot[] newDots = new Dot[dots.length]; // dots for next gen
    setBestDot();
    calculateFitnessSum();
    newDots[0] = dots[bestDot].gimmeBaby();
    newDots[0].isBest=true;
    for (int i=1; i<newDots.length; i++) {
      // select parent based on fitness
      Dot parent = selectParent();
      // get baby from them
      newDots[i] = parent.gimmeBaby();
    }
    dots = newDots;
    gen++;
  }
  // ===================================== CALCULATE FITNESS SUM =======================================================
  void calculateFitnessSum() {
    fitnessSum=0;
    for (Dot d : dots) {
      fitnessSum+=d.fitness;
    }
  }
  // The selection process is that we sum all the different fitnesses, and then chose a random number. This number 
  // will land in a "zone" i.e. greater fitness = greater probability of being chosen

  // ===================================== SELECT PARENT =======================================================
  Dot selectParent() {
    float rand = random(fitnessSum);
    float runningSum=0;

    for (Dot d : dots) {
      runningSum+=d.fitness;
      if (runningSum>rand) {
        return d;
      }
    }
    return null; // should never get to this point
  }
  // ===================================== MUTATE BABIES =======================================================
  void mutateBabies() {
    for (int i=1; i<dots.length; i++) {
      dots[i].brain.mutate();
    }
  }
  // ===================================== SET BEST DOT =======================================================
  void setBestDot() { // we take the best dot from this gen, and put it in the next gen without mutating it
    float max = 0;
    int maxIndex=0;
    for (int i=0; i<dots.length; i++) {
      if (dots[i].fitness>max) {
        max = dots[i].fitness;
        maxIndex=i;
      }
    }
    bestDot = maxIndex;
    if (dots[bestDot].reachedGoal) {
      minStep=dots[bestDot].brain.step;
      println("steps taken to get to goal:", minStep);
    }
  }
}
