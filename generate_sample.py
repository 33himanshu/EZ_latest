#!/usr/bin/env python3
"""
Generate sample documents for testing the Research Assistant.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

def generate_sample_files(output_dir: str = ".") -> None:
    """Generate sample files for testing.
    
    Args:
        output_dir: Directory where the sample files will be saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample text file
    text_content = """Research Paper: The Impact of Artificial Intelligence on Modern Society

Abstract
This paper explores the profound impact of artificial intelligence (AI) on various sectors of modern society, including healthcare, education, and the workforce. We examine both the opportunities and challenges presented by AI technologies and discuss potential future developments.

1. Introduction
Artificial Intelligence has emerged as one of the most transformative technologies of the 21st century. From virtual assistants to autonomous vehicles, AI systems are becoming increasingly integrated into our daily lives. This paper provides an overview of current AI applications and their societal implications.

2. AI in Healthcare
AI is revolutionizing healthcare through improved diagnostics, personalized treatment plans, and drug discovery. Machine learning algorithms can analyze medical images with high accuracy, often surpassing human experts in detecting conditions such as cancer and diabetic retinopathy.

3. AI in Education
In education, AI-powered platforms enable personalized learning experiences, automated grading, and intelligent tutoring systems. These technologies help bridge educational gaps and provide equal learning opportunities for students worldwide.

4. The Future of Work
While AI creates new job opportunities in tech sectors, it also poses challenges for workforce displacement. This section discusses strategies for workforce reskilling and the importance of developing AI systems that complement human capabilities.

5. Ethical Considerations
The rapid advancement of AI raises important ethical questions regarding privacy, bias, and accountability. We examine current frameworks for responsible AI development and deployment.

6. Conclusion
As AI continues to evolve, it is crucial to balance innovation with ethical considerations. Collaborative efforts between technologists, policymakers, and the public are essential to ensure that AI benefits all of humanity.

References
1. Smith, J. (2023). AI and Society: Challenges and Opportunities. Tech Press.
2. Johnson, M. (2023). The Future of Work in the Age of AI. Future Studies Journal."""
    
    # Save as text file
    text_path = output_dir / "sample.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_content)
    print(f"Created sample text file: {text_path}")
    
    # Try to create PDF if fpdf is available
    try:
        from fpdf import FPDF
        
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Research Assistant Sample Document', 0, 1, 'C')
                self.ln(10)

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(4)

            def chapter_body(self, text):
                self.set_font('Arial', '', 12)
                self.multi_cell(0, 10, text)
                self.ln()
        
        # Create PDF
        pdf = PDF()
        pdf.add_page()
        
        # Add content
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'The Impact of Artificial Intelligence on Modern Society', 0, 1, 'C')
        pdf.ln(10)
        
        # Split content into sections
        sections = text_content.split('\n\n')
        current_section = []
        
        for line in sections:
            if line.strip().endswith(':'):
                if current_section:
                    pdf.chapter_body('\n'.join(current_section).strip())
                    current_section = []
                pdf.chapter_title(line)
            else:
                current_section.append(line.strip())
        
        if current_section:
            pdf.chapter_body('\n'.join(current_section).strip())
        
        # Save PDF
        pdf_path = output_dir / "sample.pdf"
        pdf.output(str(pdf_path))
        print(f"Created sample PDF file: {pdf_path}")
        
    except ImportError:
        print("Note: Install 'fpdf' to generate PDF sample (pip install fpdf)")

def main():
    """Main function to handle command-line arguments and generate samples."""
    parser = argparse.ArgumentParser(description='Generate sample documents for testing.')
    parser.add_argument(
        '-o', '--output-dir',
        default='samples',
        help='Directory where sample files will be saved (default: samples/)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Generating sample documents...")
    generate_sample_files(args.output_dir)
    print("\n✅ Sample documents generated successfully!")

if __name__ == "__main__":
    main()
