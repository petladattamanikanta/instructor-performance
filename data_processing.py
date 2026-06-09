import os
import pandas as pd
import numpy as np

def run_preprocessing():
    print("Starting Data Preprocessing Phase...")
    
    excel_path = "data/EduPro Online Platform.xlsx"
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Source Excel file not found at {excel_path}")
        
    # Phase 1: Load all sheets
    print("Loading sheets from Excel file...")
    xls = pd.ExcelFile(excel_path)
    
    users = pd.read_excel(xls, 'Users')
    teachers = pd.read_excel(xls, 'Teachers')
    courses = pd.read_excel(xls, 'Courses')
    transactions = pd.read_excel(xls, 'Transactions')
    
    print(f"Loaded: Users ({users.shape[0]} rows), Teachers ({teachers.shape[0]} rows), Courses ({courses.shape[0]} rows), Transactions ({transactions.shape[0]} rows)")

    # Data Quality Report generation
    print("\n--- DATA QUALITY REPORT ---")
    for df_name, df in [("Users", users), ("Teachers", teachers), ("Courses", courses), ("Transactions", transactions)]:
        print(f"\nSheet: {df_name}")
        print(f"Records: {df.shape[0]} | Columns: {list(df.columns)}")
        print("Data Types:\n", df.dtypes)
        print("Missing Values:\n", df.isnull().sum())
        print("Duplicate Records Count:", df.duplicated().sum())
        
        # Outlier Detection for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            print("Outlier Scan (Values outside 1.5 * IQR):")
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                print(f"  Column '{col}': {outliers.shape[0]} outliers found (Range: [{df[col].min()}, {df[col].max()}])")

    # Phase 2: Cleaning and Preprocessing
    print("\nCleaning data...")
    # Strip whitespace from string columns
    for df in [users, teachers, courses, transactions]:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop exact duplicates if any
    users.drop_duplicates(inplace=True)
    teachers.drop_duplicates(inplace=True)
    courses.drop_duplicates(inplace=True)
    transactions.drop_duplicates(inplace=True)

    # Ensure transaction date is datetime
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

    # Feature Engineering
    print("Engineering features...")
    
    # 1. Experience Tier
    # Beginner = 0–3 years, Intermediate = 4–8 years, Experienced = 9–15 years, Expert = 15+ years
    def get_experience_tier(exp):
        if exp <= 3:
            return 'Beginner'
        elif exp <= 8:
            return 'Intermediate'
        elif exp <= 15:
            return 'Experienced'
        else:
            return 'Expert'

    teachers['ExperienceTier'] = teachers['YearsOfExperience'].apply(get_experience_tier)

    # 2. Instructor Rating Tier
    # Low <= 3.0, Medium: 3.0 - 4.0, High > 4.0
    def get_rating_tier(rating):
        if rating <= 3.0:
            return 'Low'
        elif rating <= 4.0:
            return 'Medium'
        else:
            return 'High'

    teachers['TeacherRatingTier'] = teachers['TeacherRating'].apply(get_rating_tier)

    # 3. Course Rating Tier
    courses['CourseRatingTier'] = courses['CourseRating'].apply(get_rating_tier)

    # 4. Enrollment Count per Instructor
    teacher_enrollments = transactions.groupby('TeacherID').size().reset_index(name='EnrollmentCountPerInstructor')
    teachers = teachers.merge(teacher_enrollments, on='TeacherID', how='left').fillna({'EnrollmentCountPerInstructor': 0})
    teachers['EnrollmentCountPerInstructor'] = teachers['EnrollmentCountPerInstructor'].astype(int)

    # 5. Enrollment Count per Course
    course_enrollments = transactions.groupby('CourseID').size().reset_index(name='EnrollmentCountPerCourse')
    courses = courses.merge(course_enrollments, on='CourseID', how='left').fillna({'EnrollmentCountPerCourse': 0})
    courses['EnrollmentCountPerCourse'] = courses['EnrollmentCountPerCourse'].astype(int)

    # Phase 3: Merging & Integration
    print("Integrating datasets into master analytical table...")
    
    # Prefix/Rename columns to avoid overlap
    users_renamed = users.rename(columns={
        'Age': 'UserAge',
        'Gender': 'UserGender'
    })
    
    teachers_renamed = teachers.rename(columns={
        'Age': 'TeacherAge',
        'Gender': 'TeacherGender'
    })

    # Merge sequentially
    # Start with Transactions, merge Users, Courses, Teachers
    master_df = transactions.merge(users_renamed, on='UserID', how='left')
    master_df = master_df.merge(courses, on='CourseID', how='left')
    master_df = master_df.merge(teachers_renamed, on='TeacherID', how='left')
    
    # Save the cleaned and master datasets
    os.makedirs("data", exist_ok=True)
    users.to_csv("data/cleaned_users.csv", index=False)
    teachers.to_csv("data/cleaned_teachers.csv", index=False)
    courses.to_csv("data/cleaned_courses.csv", index=False)
    transactions.to_csv("data/cleaned_transactions.csv", index=False)
    master_df.to_csv("data/master_analytical_data.csv", index=False)
    
    print("\nAll datasets processed and successfully written to 'data/' directory.")
    print(f"Master Analytical Dataset shape: {master_df.shape}")
    print("Done!")

if __name__ == "__main__":
    run_preprocessing()
