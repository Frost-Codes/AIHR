import os
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()


def get_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_data(job, cv_path):
    resume = get_text(pdf_path=cv_path)

    responsibilities = ''.join(job.responsibilities.split('\n'))
    qualifications = ''.join(job.qualifications.split('\n'))

    data = {'resume': resume, 'responsibilities': responsibilities, 'qualifications': qualifications}

    return data


def consult_ai(job, cv_path):

    data = get_data(job=job, cv_path=cv_path)

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": f"""You are an experienced HR assistant. We are shortlisting candidates for a job in the
                 company. Given a resume, job responsibilities and required qualifications you are supposed to rate that \
                 resume in a scale of 0 to 100 percent where 0 is lowest score and 100 is highest score.
                 This is is the resume: {data['resume']}
                 This are the responsibilities: {data['responsibilities']}
                 This are the qualifications: {data['qualifications']}
                    
                 return your response in a python dictionary format with score and summary as keys
                 NOTE: return dictionary output only i.e respond like and API
                 """
            },
            {
                "role": "assistant",
                "content": "" # "Java is a high-level, object-oriented programming language and platform developed by Sun Microsystems (now owned by Oracle Corporation). It was first released in 1995 and has since become one of the most popular programming languages in the world.\n\nHere are some key features of Java:\n\n**Object-Oriented**: Java is an object-oriented language, which means it organizes code into objects that contain data and behavior. This makes it easy to write reusable and modular code.\n\n**Platform Independent**: Java is designed to be platform-independent, meaning that programs written in Java can run on any device that has a Java Virtual Machine (JVM) installed. This allows Java code to run on Windows, macOS, Linux, and even mobile devices.\n\n**Simple and Familiar Syntax**: Java's syntax is based on C++ and is relatively simple to learn, especially for developers already familiar with C++ or other programming languages.\n\n**Robust Security**: Java has built-in security features, such as memory management and runtime checks, that help prevent common programming errors like null pointer exceptions and data corruption.\n\n**Large Community**: Java has a massive community of developers, which means there are many resources available for learning and troubleshooting.\n\n**Wide Range of Applications**: Java is used in a wide range of applications, including:\n\n1. **Web Development**: Java is used for developing web applications using frameworks like Spring and Hibernate.\n2. **Mobile App Development**: Java is used for developing Android apps, as well as enterprise mobile apps using Java ME.\n3. **Enterprise Software**: Java is used for developing large-scale enterprise software, such as banking and financial systems.\n4. **Desktop Applications**: Java is used for developing desktop applications, such as IDEs, media players, and games.\n5. **Machine Learning and AI**: Java is used for developing machine learning and AI applications, including natural language processing and computer vision.\n\n**Java Virtual Machine (JVM)**: The JVM is a crucial component of the Java platform. It's a virtual machine that runs Java bytecode, which is platform-independent. The JVM provides a sandboxed environment for Java code to run, ensuring that it's secure and reliable.\n\n**Java Editions**: There are several editions of Java, including:\n\n1. **Java SE** (Standard Edition): For developing desktop applications and applets.\n2. **Java EE** (Enterprise Edition): For developing enterprise-level applications, including web applications and services.\n3. **Java ME** (Micro Edition): For developing mobile applications and embedded systems.\n\nOverall, Java is a versatile, powerful, and widely-used programming language that's ideal for developing a broad range of applications."
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    data = ''

    for chunk in completion:
        try:
            data += chunk.choices[0].delta.content
        except Exception as e:
            pass

    data = data.replace("\'", "\"")

    return data

