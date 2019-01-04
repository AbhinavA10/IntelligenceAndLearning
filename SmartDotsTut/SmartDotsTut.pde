//To do: once green dot dies, it doesn't seem to switch to a better dot
/*
It could be that the dot that reaches the goal takes more steps or something, so the best dot isn't set to that.

*/
// Tutorial: https://www.youtube.com/watch?v=BOZfhUcNiqk

Population population;
Level level;
PVector goal = new PVector(350, 100); // make class for this later

// ===================================== SETUP =======================================================
void setup () {
  size(700, 700);
  population = new Population(1000);
  level = new Level();
  //frameRate(10);
}
// ===================================== DRAW =======================================================
void draw() {
  background(255);
  fill (255, 0, 0);
  ellipse(goal.x, goal.y, 10, 10);
  if (population.allDotsDead()) {
    // once all dots are dead, we start our genetic algo
    population.calculateFitness();
    population.naturalSelection();
    population.mutateBabies();
  } else {
    level.show();
    population.update();
    population.show();
  }
}
