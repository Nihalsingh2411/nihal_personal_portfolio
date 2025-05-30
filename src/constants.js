// Skills Section Logo's
import tensorflowLogo from "./assets/tech_logo/tensorflow.png";
import kerasLogo from "./assets/tech_logo/keras.png";
import pytorchLogo from "./assets/tech_logo/pytorch.png";
import sklearnLogo from "./assets/tech_logo/sklearn.png";
import huggingfaceLogo from "./assets/tech_logo/huggingface.svg";
import pandasLogo from "./assets/tech_logo/pandaslogo.png";
import excelLogo from "./assets/tech_logo/msexcel.png";
import matplotlibLogo from "./assets/tech_logo/matplotliblogo.png";
import jupyterLogo from "./assets/tech_logo/jupyter.png";
import NumpyLogo from "./assets/tech_logo/Numpy.png";
import seabornLogo from "./assets/tech_logo/seaborn.svg";
import powerbiLogo from "./assets/tech_logo/powerbi.png";
import geminiLogo from "./assets/tech_logo/gemini.png";
import flasklogo from "./assets/tech_logo/flasklogo.png";
import googlecolablogo from "./assets/tech_logo/googlecolab.png";
import fastapilogo from "./assets/tech_logo/fastapi.png";
import streamlitlogo from "./assets/tech_logo/streamlitlogo.png";
import rlogo from "./assets/tech_logo/r.jpg";
import htmlLogo from './assets/tech_logo/html.png';
import cssLogo from './assets/tech_logo/css.png';
import javascriptLogo from './assets/tech_logo/javascript.png';
import mysqlLogo from './assets/tech_logo/mysql.png';
import mongodbLogo from './assets/tech_logo/mongodb.png';
import cLogo from './assets/tech_logo/c.png';
import cppLogo from './assets/tech_logo/cpp.png';
import pythonLogo from './assets/tech_logo/python.png';
import gitLogo from './assets/tech_logo/git.png';
import githubLogo from './assets/tech_logo/github.png';
import vscodeLogo from './assets/tech_logo/vscode.png';
import postmanLogo from './assets/tech_logo/postman.png';
import vercelLogo from './assets/tech_logo/vercel.png';
import langchainLogo from './assets/tech_logo/langchain.svg'; 
import openCVLogo from './assets/tech_logo/opencv.png'; 

// Experience Section Logo's
import corecardLogo from './assets/company_logo/corecard.png';
import mitsLogo from './assets/company_logo/mits.jpg';

// Education Section Logo's

import rishigalavLogo from './assets/education_logo/rishiglv.jpg';
import sjvsLogo from './assets/education_logo/sjvs.png';

// Project Section Logo's

import resumeLogo from './assets/work_logo/resumeAnalyzer.webp';
import networkLogo from './assets/work_logo/networksecurity2.jpg';
import imgcapLogo from './assets/work_logo/imgcaptioning.png';
import churnLogo from './assets/work_logo/churn.webp';
import chestxrayLogo from './assets/work_logo/chestxray.webp';
import moviestreamLogo from './assets/work_logo/moviestream.jpg';
import portfolioLogo from './assets/work_logo/portfolio.png';

export const SkillsInfo = [
  {
    title: 'Data Science',
    skills: [
      { name: 'Python', logo: pythonLogo },
      { name: 'Pandas', logo: pandasLogo },
      { name: 'Numpy', logo: NumpyLogo },
      { name: 'Matplotlib', logo: matplotlibLogo },
      { name: 'Seaborn', logo: seabornLogo },
      { name: 'Jupyter Notebook', logo: jupyterLogo },
      { name: 'Excel', logo:excelLogo},
      { name: 'PowerBI', logo: powerbiLogo },
      
    ],
  },
  {
    title: 'AI & Machine Learning',
    skills: [
      { name: 'Tensorflow', logo: tensorflowLogo },
      { name: 'Keras', logo: kerasLogo },
      { name: 'Pytorch', logo: pytorchLogo },
      { name: 'Scikit-learn', logo: sklearnLogo },
      { name: 'OpenCV', logo: openCVLogo},
      { name: 'Hugging Face', logo: huggingfaceLogo },
      { name: 'LangChain', logo: langchainLogo },
      { name: 'GeminiAPI', logo: geminiLogo },
      { name: 'Flask', logo: flasklogo },
    ],
  },
  {
    title: 'Languages',
    skills: [
      { name: 'C', logo: cLogo },
      { name: 'C++', logo: cppLogo },
      { name: 'Python', logo: pythonLogo },
      { name: 'JavaScript', logo: javascriptLogo },
      { name: 'R', logo: rlogo },
      { name: 'MySQL', logo: mysqlLogo },
      { name: 'HTML', logo: htmlLogo },
      { name: 'CSS', logo: cssLogo },
      
    ],
  },
  {
    title: 'Tools',
    skills: [
      { name: 'Git', logo: gitLogo },
      { name: 'GitHub', logo: githubLogo },
      { name: 'VS Code', logo: vscodeLogo },
      { name: 'Postman', logo: postmanLogo },
      { name: 'Google colab', logo: googlecolablogo},
      { name: 'Vercel', logo: vercelLogo },
      { name: 'Streamlit', logo: streamlitlogo },
      { name: 'FastAPI', logo: fastapilogo },
      { name: 'MongoDB', logo: mongodbLogo },
    ],
  },
];

  export const experiences = [
    {
      id: 0,
      img: corecardLogo,
      role: "Software Engineer Intern",
      company: "Corecard Software India Pvt. Ltd. Bhopal",
      date: " Jun 2024 - Jul 2024",
      desc: "Rebuilt the company website using the MERN stack, achieving 30% faster load times and enhanced performance. Developed 5+ responsive UI components and resolved 30+ front-end bugs to boost stability. Collaborated in design reviews to improve overall user experience.",
      skills: [
        "HTML",
        "CSS",
        "Bootstrap",
        "JavaScript",
        "React.js",
        "UI/UX",
        "REST API",
        "Postman",
        "A/B Testing",
        "Font Awesome",
      ],
    },
    {
      id: 1,
      img: mitsLogo,
      role: "Undergraduate Summer Intern",
      company: "MITS Gwalior",
      date: "Jun 2023 - Jul 2023",
      desc: "Preprocessed a dataset of 10,000 records, resolving missing values and outliers to ensure data quality. Conducted in-depth exploratory data analysis and built a machine learning model achieving 95% accuracy for reliable predictions.",
      skills: [
        "Python",
        "Pandas",
        "Numpy",
        "Matplotlib",
        "Seaborn",
        "EDA",
        "Visualization",
        "Regression",
      ],
    },
  ];
  
  export const education = [
    {
      id: 0,
      img: mitsLogo,
      school: "Madhav Institute of Technology and Science, Gwalior",
      date: "2022-2026",
      grade: "8.45 GPA",
      desc: "I am currently pursuing B.Tech in Artificial Intelligence and Machine Learning from Madhav Institute of Technology and Science (MITS), Gwalior, with a current GPA of 8.45. I have gained a strong foundation in subjects like Data Structures and Algorithms, OOPS, Operating Systems, DBMS, Computer Networks and Data Science. I have been actively involved in academic and technical activities.This ongoing journey at MITS is continuously shaping my capabilities in the field of AI and computer science.",
      degree: "B.Tech in Artificial Intelligence & Machine Learning",
    },
    {
      id: 1,
      img: rishigalavLogo,
      school: "Rishi Galav Public School, Gwalior",
      date: "Apr 2020 - March 2021",
      grade: "92%",
      desc: "I completed my class 12 education from Rishi Galav Public School, Gwalior, under the CBSE board, where I studied Physics, Chemistry, and Mathematics (PCM), sparking my interest in technology and programming.",
      degree: "CBSE(XII) - PCM",
    },
    {
      id: 2,
      img: sjvsLogo,
      school: "St.John Vianney School,Gwalior",
      date: "Apr 2018 - March 2019",
      grade: "90%",
      desc: "I completed my class 10 education from Vatsalya Public School, Govardhan, under the CBSE board, where I studied Science with Information Technology.",
      degree: "CBSE(X), Science with Information Technology",
    },
  ];
  
  export const projects = [
    {
      id: 0,
      title: "AI Resume Analyzer",
      description:
        `An intelligent resume analyzer that uses Gemini to assess ATS compatibility, suggest improvements, and generate tailored cover letters. Includes a dynamic dashboard for managing uploads, tracking classifications, and exploring relevant job listings—streamlining the path to resume shortlisting.`,
      image: resumeLogo,
      tags: ["Python","NLP", "Gen AI", "SQL","Google Gemini", "Rapid API","LLM","Name-Entity Recognition","Streamlit"],
      github: "https://github.com/2411nihalsingh/NLP-Based-Resume-Analyzer",
      webapp: "https://ai-resume-analyzer-hbdwxwbfabmgrvfkrklmqs.streamlit.app/",
    },
    {
      id: 1,
      title: "MLOps Pipeline for Network Security System",
      description:"A robust end-to-end MLOps pipeline for network security, featuring automated data processing, model training, and evaluation. Integrated with MongoDB and MLflow for seamless tracking, and deployed on AWS using Docker and FastAPI for scalable, real-time threat classification",
      image: networkLogo,
      tags: ["Machine Learning","Python", "Classification", "Random Forest" ,"AWS","CI/CD","MLflow"],
      github: "https://github.com/2411nihalsingh/networksecurity",
      webapp: "https://csprep.netlify.app/",
    },
    {
      id: 2,
      title: "Image Caption Generator",
      description: "A React-based web application that provides movie recommendations based on different criteria, such as genres, user preferences, and popular trends. The intuitive design and smooth experience make it a go-to app for movie enthusiasts.",
      image: imgcapLogo,
      tags: ["Python", "Deep Learning","CNN", "NLP","Transfer Learnning", "Kaggle"],
      github: "https://github.com/2411nihalsingh/Image_Captioning_Project_Using_CNN_NLP",
      webapp: "",
    },
    {
      id: 3,
      title: "Customer Churn Prediction",
      description:
        "A deep learning-based customer churn prediction model using artificial neural networks (ANN). It analyzes customer behavior to identify churn risk and helps businesses improve retention through data-driven strategies.",
      image: churnLogo,
      tags: ["ANN", "Tensorflow", "Numpy", "Scikit-learn","Tensorboard"],
      github: "https://github.com/2411nihalsingh/ANN_Classification_Churn",
      webapp: "",
    },
    {
      id: 4,
      title: "Chest X-Ray Classifier",
      description:
        "An intelligent deep learning system designed to classify chest X-rays into Normal, Pneumonia, or COVID-19 categories using Convolutional Neural Networks (CNNs). The model is trained on a labeled medical imaging dataset to achieve high diagnostic accuracy. Integrated with OpenCV, the system allows real-time image capture and instant classification, making it suitable for rapid preliminary screening in clinical settings. The project demonstrates the potential of AI in assisting radiologists by automating and accelerating diagnosis processes.",
      image: chestxrayLogo,
      tags: ["CNN", "Deep Learning", "OpenCV", "AI","Bio Medical","Streamlit","Feature Extraction"],
      github: "https://github.com/2411nihalsingh/Chest_X-Ray_Classification_Using_CNN",
      webapp: "",
    },
    {
      id: 5,
      title: "Movies Streaming Platform",
      description:
        "A full-stack media platform with user login, movie recommendations, and genre-wise browsing. Built using React.js and Node.js, it features a responsive UI, secure backend, and personalized content suggestions for an engaging viewing experience.",
      image: moviestreamLogo,
      tags: ["React.js", "Node.js", "MongoDB", "Express", "JWT","Full Stack","TMDB API"],
      github: "https://github.com/2411nihalsingh/Netflix-Clone",
      webapp: "",
    },
    {
      id: 6,
      title: "Portfolio Website",
      description:
        "A sleek and responsive React.js portfolio website showcasing my projects, skills, and experience. Built with Tailwind CSS, it features smooth navigation, dark mode, and interactive UI components—designed to offer recruiters and visitors a seamless browsing experience.",
      image: portfolioLogo,
      tags: ["HTML", "CSS", "JavaScript", "Tailwind","React.js","EmailJS","Responsive"],
      github: "https://github.com/codingmastr/Webverse-Digital",
      webapp: "",
    },
  ];  