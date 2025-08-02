import pandas as pd
import random

# Define project names
projects = ["ACME", "GammaTech", "BetaCorp", "DeltaInc", "OmegaSoft"]

# Define sprint names
sprints = [f"Sprint {i}" for i in range(1, 8)]

# Define ticket title templates with more variety
ticket_templates = [
    "Backend - {action} {component}",
    "Frontend - {action} {component}",
    "Mobile - {action} {component}",
    "DevOps - {action} {component}",
    "API - {action} {component}",
    "Bug Fix - {issue} in {area}"
]

actions = [
    "Implement", "Optimize", "Refactor", "Update", "Create", 
    "Integrate", "Fix", "Enhance", "Develop", "Automate"
]

components = [
    "authentication system", "database queries", "user interface", 
    "notification service", "payment gateway", "reporting module",
    "caching mechanism", "error handling", "data validation",
    "search functionality", "export feature", "admin dashboard"
]

issues = [
    "UI glitches", "performance bottlenecks", "memory leaks", 
    "incorrect calculations", "data inconsistencies", "broken links",
    "security vulnerabilities", "timeout errors", "validation errors"
]

areas = [
    "checkout process", "login flow", "dashboard", "mobile app", 
    "reporting system", "user profile", "admin panel", "search results"
]

priorities = ["Low", "Medium", "High", "Highest"]
dependencies = ["-", "API Integration", "Database Update", "Frontend Change", "Backend Logic Update"]

# Generate 200 records
data = []
for _ in range(200):
    ticket_template = random.choice(ticket_templates)
    
    if "Backend" in ticket_template or "Frontend" in ticket_template or "Mobile" in ticket_template or "DevOps" in ticket_template or "API" in ticket_template:
        ticket = ticket_template.format(action=random.choice(actions), component=random.choice(components))
    else:
        ticket = ticket_template.format(issue=random.choice(issues), area=random.choice(areas))
    
    # Story points are typically 1, 2, 3, 5, 8 in Fibonacci sequence
    story_point = random.choice([1, 2, 3, 5])
    
    # Deadline should be reasonable based on story points
    deadline = random.randint(60, 120) * story_point
    
    # Duration should be somewhat related to deadline but with variability
    duration_factor = random.uniform(0.7, 1.3)
    duration = int(deadline * duration_factor)
    
    data.append({
        "PROJECT NAME (ANONYMISED)": random.choice(projects),
        "SPRINT NAME": random.choice(sprints),
        "TICKET TITLE": ticket,
        "DEADLINE (MINUTES)": deadline,
        "DURATION (MINUTES)": duration,
        "STORY POINT": story_point,
        "PRIORITY": random.choice(priorities),
        "DEPENDENCY": random.choice(dependencies)
    })

# Create DataFrame
synthetic_data = pd.DataFrame(data)

# Save to CSV
synthetic_data.to_csv("Synthetic_Agile_Project_Data.csv", index=False)

# Display first few rows
synthetic_data.head()
