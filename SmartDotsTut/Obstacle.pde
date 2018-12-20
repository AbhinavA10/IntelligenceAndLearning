class Obstacle {
  // class for obstacles in population's way
  PVector pos;
  int boxWidth, boxHeight;
  //PVector[] verticies = new  PVector[4];
  // ===================================== OBSTACLE CONSTRUCTOR =======================================================
  Obstacle(int x, int y, int w, int h){
    pos = new PVector(x, y);
    boxWidth=w;
    boxHeight=h;
    //verticies[0] =  pos;
    // find other corner points
  }
  // ===================================== SHOW =======================================================
  void show(){
    fill (0,0,255);
    rect(pos.x,pos.y,boxWidth,boxHeight);
  }
  
}
