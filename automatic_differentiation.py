# Automatic differentiation in reverse (Backpropagation) algorithm

class Node:
    def __init__(self, value, grad=0.0):
        self.value = value
        self.grad = grad
        self.parents = []
        self.id = id
    
    def add_parent(self, parent):
        self.parents.append(parent)

def add(n1, n2):
    result = Node(n1.value + n2.value)
    result.add_parent((n1, 1.0))
    result.add_parent((n2, 1.0))
    return result

def mul_nodes(n1, n2):
    result = Node(n1.value * n2.value)
    result.add_parent((n1, n2.value))
    result.add_parent((n2, n1.value))
    return result

def mul_scalar(n, scalar):
    result = Node(n.value * scalar)
    result.add_parent((n, scalar))
    return result

def sqrt(n):
    result = Node(n.value ** 0.5)
    result.add_parent((n, 1 / (2 * result.value)))
    return result

def square(n):
    result = Node(n.value ** 2)
    result.add_parent((n, 2*n.value))
    return result

def backpropagate(node, grad=1, child_id = "f(x)"):
    node.grad += grad
    print(f"{node.id}.grad += d{node.id} /d{child_id} = ", node.grad)
    for parent, parent_grad in node.parents:
        backpropagate(parent, grad * parent_grad, node.id)

# f(x) = x^{2} + 2x
node1 = Node(2)
node2 = square(node1)
node3 = mul_scalar(node1, 2)
node4 = add(node2, node3)

# df/dx
backpropagate(node5)
print(node1.grad)   

