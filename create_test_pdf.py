"""Create a simple text-based PDF for testing the research assistant."""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_test_pdf(output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Page 1: Introduction to Machine Learning
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, 10*inch, "Introduction to Machine Learning")
    
    c.setFont("Helvetica", 12)
    text = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn",
        "and improve from experience without being explicitly programmed. It focuses on the",
        "development of computer programs that can access data and use it to learn for themselves.",
        "",
        "The main types of machine learning are:",
        "",
        "1. Supervised Learning: The algorithm learns from labeled training data and makes",
        "   predictions based on that data. Examples include classification and regression.",
        "",
        "2. Unsupervised Learning: The algorithm finds patterns in unlabeled data without",
        "   explicit guidance. Examples include clustering and dimensionality reduction.",
        "",
        "3. Reinforcement Learning: The algorithm learns by interacting with an environment",
        "   and receiving rewards or penalties for actions taken.",
    ]
    
    y_position = 9*inch
    for line in text:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.2*inch
    
    c.showPage()
    
    # Page 2: Common Algorithms
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1*inch, 10*inch, "Common Machine Learning Algorithms")
    
    c.setFont("Helvetica", 12)
    algorithms = [
        "Linear Regression: Used for predicting continuous values based on input features.",
        "It finds the best-fitting linear relationship between variables.",
        "",
        "Decision Trees: A flowchart-like structure where each internal node represents a",
        "test on an attribute, each branch represents the outcome, and each leaf node",
        "holds a class label or value.",
        "",
        "Random Forest: An ensemble method that builds multiple decision trees and",
        "combines their predictions for improved accuracy and stability.",
        "",
        "Support Vector Machines (SVM): Finds the optimal hyperplane that separates",
        "data points of different classes with the maximum margin.",
        "",
        "Neural Networks: Inspired by biological neurons, these consist of interconnected",
        "nodes organized in layers that can learn complex non-linear relationships.",
    ]
    
    y_position = 9*inch
    for line in algorithms:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.2*inch
    
    c.save()
    print(f"Created test PDF: {output_path}")

if __name__ == "__main__":
    output_path = "research_assistant/test_pdfs/ml_basics.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_test_pdf(output_path)
