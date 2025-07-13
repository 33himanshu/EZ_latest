#!/usr/bin/env python3
"""
Create a sample PDF document for testing the Research Assistant.
"""
from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        # Add a logo or title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Research Assistant Sample Document', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        # Format chapter titles
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        # Format chapter body
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln()

def create_sample_pdf(output_path: str = "sample.pdf"):
    """Create a sample PDF document.
    
    Args:
        output_path: Path where the PDF will be saved
    """
    # Create PDF object
    pdf = PDF()
    pdf.add_page()
    
    # Add content
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'The Impact of Artificial Intelligence on Modern Society', 0, 1, 'C')
    pdf.ln(10)
    
    # Abstract
    pdf.chapter_title('Abstract')
    abstract = """This paper explores the profound impact of artificial intelligence (AI) on various 
    sectors of modern society, including healthcare, education, and the workforce. We examine both 
    the opportunities and challenges presented by AI technologies and discuss potential future 
    developments."""
    pdf.chapter_body(abstract)
    
    # Introduction
    pdf.chapter_title('1. Introduction')
    intro = """Artificial Intelligence has emerged as one of the most transformative technologies 
    of the 21st century. From virtual assistants to autonomous vehicles, AI systems are becoming 
    increasingly integrated into our daily lives. This paper provides an overview of current AI 
    applications and their societal implications."""
    pdf.chapter_body(intro)
    
    # AI in Healthcare
    pdf.chapter_title('2. AI in Healthcare')
    healthcare = """AI is revolutionizing healthcare through improved diagnostics, personalized 
    treatment plans, and drug discovery. Machine learning algorithms can analyze medical images 
    with high accuracy, often surpassing human experts in detecting conditions such as cancer and 
    diabetic retinopathy."""
    pdf.chapter_body(healthcare)
    
    # AI in Education
    pdf.chapter_title('3. AI in Education')
    education = """In education, AI-powered platforms enable personalized learning experiences, 
    automated grading, and intelligent tutoring systems. These technologies help bridge educational 
    gaps and provide equal learning opportunities for students worldwide."""
    pdf.chapter_body(education)
    
    # The Future of Work
    pdf.chapter_title('4. The Future of Work')
    work = """While AI creates new job opportunities in tech sectors, it also poses challenges 
    for workforce displacement. This section discusses strategies for workforce reskilling and the 
    importance of developing AI systems that complement human capabilities."""
    pdf.chapter_body(work)
    
    # Ethical Considerations
    pdf.chapter_title('5. Ethical Considerations')
    ethics = """The rapid advancement of AI raises important ethical questions regarding privacy, 
    bias, and accountability. We examine current frameworks for responsible AI development and 
    deployment."""
    pdf.chapter_body(ethics)
    
    # Conclusion
    pdf.chapter_title('6. Conclusion')
    conclusion = """As AI continues to evolve, it is crucial to balance innovation with ethical 
    considerations. Collaborative efforts between technologists, policymakers, and the public are 
    essential to ensure that AI benefits all of humanity."""
    pdf.chapter_body(conclusion)
    
    # References
    pdf.chapter_title('References')
    references = """1. Smith, J. (2023). AI and Society: Challenges and Opportunities. Tech Press.
    2. Johnson, M. (2023). The Future of Work in the Age of AI. Future Studies Journal."""
    pdf.chapter_body(references)
    
    # Save the PDF
    pdf.output(output_path)
    print(f"Sample PDF created at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_sample_pdf("sample.pdf")
