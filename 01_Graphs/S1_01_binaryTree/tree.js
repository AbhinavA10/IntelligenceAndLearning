
function Tree() { // object
  this.root; //root node;
}

Tree.prototype.addValue = function(val) { //creating a function in tree object
  var node = new Node(val);
  if (this.root == null) {
    this.root = node;
  } else {
    this.root.addNode(node);
  }

}
Tree.prototype.traverse = function() {
  root.visit
}
