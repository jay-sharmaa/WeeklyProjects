import ast
import os
import io
import graphviz
from flask import Flask, request, send_file

class SimpleASTGraph(ast.NodeVisitor):
    def __init__(self):
        self.graph = graphviz.Digraph(format="png")
        self.graph.attr(dpi="300")
        self.node_count = 0

    def new_node(self, label):
        node_id = f"node{self.node_count}"
        self.node_count += 1
        self.graph.node(node_id, label=label, shape="box", style="filled", color="lightblue")
        return node_id

    def visit_Assign(self, node):
        assign_node = self.new_node("Assign")

        for target in node.targets:
            target_node = self.visit(target)
            self.graph.edge(assign_node, target_node)

        value_node = self.visit(node.value)
        self.graph.edge(assign_node, value_node)

        return assign_node

    def visit_Name(self, node):
        return self.new_node(f"Name: {node.id}")

    def visit_Constant(self, node):
        return self.new_node(f"Const: {repr(node.value)}")

    def visit_FunctionDef(self, node):
        func_node = self.new_node(f"Function: {node.name}")
        for stmt in node.body:
            child_id = self.visit(stmt)
            if child_id:
                self.graph.edge(func_node, child_id)
        return func_node

    def visit_If(self, node):
        if_node = self.new_node("If")
        test_id = self.visit(node.test)
        if test_id:
            self.graph.edge(if_node, test_id)
        for stmt in node.body:
            stmt_id = self.visit(stmt)
            if stmt_id:
                self.graph.edge(if_node, stmt_id)
        for stmt in node.orelse:
            stmt_id = self.visit(stmt)
            if stmt_id:
                self.graph.edge(if_node, stmt_id)
        return if_node

    def visit_Return(self, node):
        return_node = self.new_node("Return")
        value_id = self.visit(node.value)
        if value_id:
            self.graph.edge(return_node, value_id)
        return return_node

    def visit_BinOp(self, node):
        op_node = self.new_node(f"BinOp: {node.op.__class__.__name__}")
        left_id = self.visit(node.left)
        right_id = self.visit(node.right)
        if left_id:
            self.graph.edge(op_node, left_id)
        if right_id:
            self.graph.edge(op_node, right_id)
        return op_node

    def visit_Compare(self, node):
        comp_node = self.new_node(f"Compare: {node.ops[0].__class__.__name__}")
        left_id = self.visit(node.left)
        if left_id:
            self.graph.edge(comp_node, left_id)
        for comparator in node.comparators:
            right_id = self.visit(comparator)
            if right_id:
                self.graph.edge(comp_node, right_id)
        return comp_node

    def generic_visit(self, node):
      return None


    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            call_node = self.new_node(f"Call: {node.func.id}")
        else:
            call_node = self.new_node("Call")

        for arg in node.args:
            arg_node = self.visit(arg)
            self.graph.edge(call_node, arg_node)

        return call_node

    def visit_Expr(self, node):
        return self.visit(node.value)

    def generate_graph(self, code):
        tree = ast.parse(code)
        for stmt in tree.body:
            self.visit(stmt)
        return self.graph

def generate_ast_graph(code, output_filename="simple_ast"):
    graph_generator = SimpleASTGraph()
    graph = graph_generator.generate_graph(code)
    image_path = graph.render(output_filename, format="png", cleanup=True)
    with open(image_path, "rb") as f:
        return f.read()

app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def index():
    return "AST Visualization API. Send POST requests with JSON body containing 'code' parameter."

@app.route("/", methods=["POST"])
def handle_code():
    code = request.json.get("code")
    image_bytes = generate_ast_graph(code)
    return send_file(io.BytesIO(image_bytes), mimetype="image/png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)