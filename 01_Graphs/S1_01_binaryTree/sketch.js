//https://www.youtube.com/watch?v=ZNH0MuQ51m4&list=PLRqwX-V7Uu6YJ3XfHhT2Mm4Y5I99nrIKX&index=3
var tree //new tree object

function setup() {
  noCanvas();
  tree = new Tree();
  tree.addValue(5);
  tree.addValue(4);
  tree.addValue(3);
  tree.addValue(6);
  console.log(tree);
  tree.traverse();
}
