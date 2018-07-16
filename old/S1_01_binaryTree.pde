//https://www.youtube.com/watch?v=ZNH0MuQ51m4&list=PLRqwX-V7Uu6YJ3XfHhT2Mm4Y5I99nrIKX&index=3
Tree tree;
void setup() {
  tree = new Tree();
  Node n = new Node(3);
  tree.addNode(n);
  println(tree);
}
