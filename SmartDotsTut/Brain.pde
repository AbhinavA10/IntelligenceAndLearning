class Brain {
  PVector[] directions; // series of pvectors used to set acceleration
  int step =0;
  // ===================================== BRAIN CONSTRUCTOR =======================================================
  Brain(int size) {
    directions = new PVector[size];
    randomize();
  }
  // ===================================== RANDOMIZE =======================================================
  void randomize() {
    for (int i=0; i<directions.length; i++) {
      float randomAngle = random(2*PI);
      directions[i] = PVector.fromAngle(randomAngle);
    }
  }
  // ===================================== CLONE =======================================================
  Brain clone() {
    Brain clone = new Brain(directions.length);
    for (int i=0; i< directions.length; i++) {
      clone.directions[i] = directions[i].copy(); // each child's brain, is exaclty the same as its parent's
    }
    return clone;
  }
  // ===================================== MUTATE =======================================================
  void mutate() {
    float mutationRate=0.01;
    for (int i=0; i<directions.length; i++) {
      float rand = random(1);
      if (rand<mutationRate) { // 1% of the directions will be mutated
        //set this direction as a random direction
        float randomAngle = random(2*PI);
        directions[i] = PVector.fromAngle(randomAngle);
      }
    }
  }
}
