"""config for data generation and preprocessing."""

config = {
    "ol_subjects": ["Maths", "Science", "English", "Sinhala", "History", "Religion"],
    "al_streams": {
        "Physical Science": ["Physics", "Chemistry", "Combined Maths"],
        "Biological Science": ["Biology", "Chemistry", "Physics/Agriculture"],
        "Commerce": ["Accounting", "Business Studies", "Economics"],
        "Arts": ["History", "Geography", "Logic/Political Science/Language"],
        "Technology": [
            "Engineering Technology",
            "Science for Technology",
            "Information & Communication Technology",
        ],
    },
    "university_courses": {
        "Physical Science": [
            "Engineering",
            "Physical Science",
            "Computer Science",
            "Architecture",
        ],
        "Biological Science": [
            "Medicine",
            "Dentistry",
            "Veterinary Medicine",
            "Biological Science",
            "Agriculture",
        ],
        "Commerce": [
            "Commerce/Management",
            "Business Administration",
            "Finance",
            "Accounting",
        ],
        "Arts": ["Arts", "Law", "Social Sciences", "Languages"],
        "Technology": [
            "Engineering Technology",
            "Information Technology",
            "Quantity Surveying",
        ],
    },
    "career_paths": {
        "Engineering": [
            "Civil Engineer",
            "Mechanical Engineer",
            "Electrical Engineer",
            "Software Engineer",
        ],
        "Physical Science": ["Physicist", "Chemist", "Data Scientist", "Researcher"],
        "Computer Science": [
            "Software Developer",
            "Data Analyst",
            "Cybersecurity Analyst",
            "AI Engineer",
        ],
        "Architecture": ["Architect", "Urban Planner", "Interior Designer"],
        "Medicine": ["Doctor", "Surgeon", "Specialist"],
        "Dentistry": ["Dentist"],
        "Veterinary Medicine": ["Veterinarian"],
        "Biological Science": ["Biologist", "Zoologist", "Botanist"],
        "Agriculture": ["Agricultural Scientist", "Farmer", "Food Scientist"],
        "Commerce/Management": ["Manager", "Consultant", "Entrepreneur"],
        "Business Administration": [
            "Business Analyst",
            "Marketing Manager",
            "HR Manager",
        ],
        "Finance": ["Financial Analyst", "Accountant", "Investment Banker"],
        "Accounting": ["Accountant", "Auditor", "Financial Controller"],
        "Arts": ["Teacher", "Historian", "Writer", "Journalist"],
        "Law": ["Lawyer", "Judge", "Legal Consultant"],
        "Social Sciences": ["Sociologist", "Psychologist", "Economist"],
        "Languages": ["Translator", "Interpreter", "Linguist"],
        "Engineering Technology": ["Technician", "Technologist", "Project Manager"],
        "Information Technology": [
            "IT Specialist",
            "Network Administrator",
            "System Analyst",
        ],
        "Quantity Surveying": ["Quantity Surveyor"],
    },
    "num_students": 1000,
    "data_dir": "./data/raw",
}
