class Dot {
  PVector pos, vel, acc;
  int radius=0;
  Brain brain;
  boolean dead=false;
  boolean reachedGoal=false;
  boolean isBest=false;
  float fitness = 0; // to figure out which dots are the best
  // ===================================== DOT CONSTRUCTOR =======================================================
  Dot() {
    brain = new Brain(400); // give the dot a brain, with 400 vectors
    pos = new PVector(width/2, height-10);
    vel = new PVector(0, 0);
    acc = new PVector(0, 0);
    radius=2;
  }
  // ===================================== SHOW =======================================================
  void show() {
    if (isBest) { // showing the champion dot to stand out
      fill(0, 255, 0);
      radius=4; // make the best dot bigger
    } else {
      fill(0);
    }
    ellipse (pos.x, pos.y, radius*2, radius*2);
  }
// ===================================== MOVE =======================================================
  void move() {
    if (brain.directions.length>brain.step) {
      acc = brain.directions[brain.step]; // set dot's current acceleration to the vector in the brain array
      brain.step++;
    } else {
      dead = true;
    }
    vel.add(acc);
    vel.limit(5); // to stop from forever accelerating
    pos.add(vel);
  }
  // ===================================== UPDATE =======================================================
  void update() {
    if (!dead && !reachedGoal) {
      move(); 
      if (pos.x< 2||pos.y<2||pos.x>width-2||pos.y>height- 2) { // top left corner of screen is (0,0)
        dead=true;
      } else if (dist(pos.x, pos.y, goal.x, goal.y)<5) {
        reachedGoal=true;
      } else {
        level.checkCollisions(this);
      }
    }
  }
  // ===================================== CALCULATE FITNESS =======================================================
  void calculateFitness() {
    if (reachedGoal) {// to get dots to goal in shortest number of steps possible
      fitness=1.0/16.0 + 10000.0/(float)(brain.step*brain.step); // all dots that reach the goal have a better fitness than those that dont
    } else {
      float distanceToGoal =dist(pos.x, pos.y, goal.x, goal.y); // the smaller this is, the better
      fitness = 1/(distanceToGoal*distanceToGoal); // squaring magnifies small distances
    }
  }

  // to clone/make a baby
  // ===================================== GIMME BABY =======================================================
  Dot gimmeBaby() {
    Dot baby = new Dot();
    baby.brain = brain.clone();
    return baby;
  }
}
