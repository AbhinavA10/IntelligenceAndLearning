// For collision detecion https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
//http://jeffreythompson.org/collision-detection/circle-rect.php
class Level {
  // array list of obstacles
  ArrayList<Obstacle> obstacleList;
  // ===================================== LEVEL CONSTRUCTOR =======================================================
  Level() {
    obstacleList = new ArrayList<Obstacle>();  
    obstacleList.add(new Obstacle(200, 200, 200, 10));
    obstacleList.add(new Obstacle(30,500, 500, 10));
    obstacleList.add(new Obstacle(500,350, 200, 20));
    // add obstacles here
  }
  // ===================================== SHOW =======================================================
  void show() {
    for (Obstacle obst : obstacleList) {
      obst.show();
    }
  }
  // ===================================== CHECK COLLISIONS =======================================================
  void checkCollisions(Dot d) {
    for (Obstacle obst : obstacleList) {
      if (isHit(d, obst)) {
        d.dead = true;
      }
    }
  }
  // ===================================== IS HIT =======================================================

  // change this to rotate rectangles too later
  boolean isHit(Dot dot, Obstacle obstac) {

    // temporary variables to set edges for testing
    float testX = dot.pos.x;
    float testY = dot.pos.y;

    // which edge is closest?
    if (dot.pos.x < obstac.pos.x)
      testX = obstac.pos.x;      // test left edge
    else if (dot.pos.x > obstac.pos.x+obstac.boxWidth)
      testX = obstac.pos.x+obstac.boxWidth;   // right edge
    if (dot.pos.y < obstac.pos.y)        
      testY = obstac.pos.y;      // top edge
    else if (dot.pos.y > obstac.pos.y+obstac.boxHeight) 
      testY = obstac.pos.y+obstac.boxHeight;   // bottom edge

    // get distance from closest edges
    float distX = dot.pos.x-testX;
    float distY = dot.pos.y-testY;
    float distanceSquared = (distX*distX) + (distY*distY);
    float radiusSquared = dot.radius*dot.radius;

    // if the distance is less than the radius, collision!
    if (distanceSquared <= radiusSquared) {
      return true;
    }
    return false;
  }
}
