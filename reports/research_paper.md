# Instructor Performance and Course Quality Evaluation on EduPro: A Data-Driven Approach to Learning Platform Optimization

**Role:** Senior Data Analyst & Data Scientist  
**Date:** June 3, 2026  
**Institution:** EduPro Analytics Group  

---

## 1. Abstract
This paper presents an end-to-end analytical and statistical evaluation of instructor effectiveness and course quality on the EduPro online learning platform. Utilizing a dataset containing 3,000 students, 60 instructors, 60 courses, and 10,000 transaction records, we investigate the determinants of student satisfaction, enrollment patterns, and pedagogical quality. Our findings reveal a massive structural imbalance: two veteran instructors account for over 60% of all platform enrollments. Furthermore, statistical analysis confirms that while teaching experience is highly correlated with instructor ratings ($\rho = 0.598, p < 0.001$), there is no significant relationship between teacher quality and course quality ($r = 0.0003, p = 0.998$). We segment the instructor cohort into four distinct tiers using K-Means clustering: *Star Performers*, *Average Instructors*, *Growing Instructors*, and *At-Risk Instructors*. We propose a series of strategic interventions to diversify enrollment distribution, standardize course quality, and optimize teacher allocation.

---

## 2. Introduction
Online education platforms have experienced exponential growth, necessitating rigorous mechanisms to evaluate educational quality and operational efficiency. EduPro, a premier digital learning provider, offers courses across 12 distinct domains. To maintain competitive advantage, the platform must understand how its human capital (instructors) and curricular assets (courses) interact to drive student enrollment and retention.

This study aims to answer the following research questions:
1. Does teaching experience translate to higher student ratings for instructors and courses?
2. Is there a synergy between teacher ratings and course content ratings?
3. What are the key drivers of enrollment volume on the platform?
4. How can the instructor pool be segmented to tailor professional development and incentives?

---

## 3. Problem Statement
EduPro is currently operating in an information vacuum regarding instructor distribution and enrollment dynamics. A preliminary inspection of the transactional data suggests that enrollment is highly concentrated, but the exact drivers remain unquantified. Furthermore, the platform lacks integrated Key Performance Indicators (KPIs) to monitor quality, leading to sub-optimal decisions in hiring, training, and course development. 

---

## 4. Methodology
Our analysis follows a rigorous, five-stage methodology:
1. **Data Preprocessing & Cleaning**: Cleaned string entries, checked for null values and duplicates, and parsed dates. We engineered derived metrics such as *Experience Tiers* (Beginner: 0-3 yrs, Intermediate: 4-8 yrs, Experienced: 9-15 yrs, Expert: 15+ yrs), *Rating Tiers* (Low: $\le 3.0$, Medium: 3.0-4.0, High: $> 4.0$), and teacher-specific and course-specific total enrollment counts.
2. **Data Integration**: Joined transactional data with user, course, and instructor dimension tables to build a unified analytical master file.
3. **Exploratory Data Analysis (EDA)**: Analyzed demographic and categorical distributions across the platform.
4. **Statistical Correlation**: Applied Pearson (linear) and Spearman (rank-based) correlation coefficients to evaluate the relationships between experience, ratings, age, and enrollments, verifying significance using p-values.
5. **Instructor Segmentation (K-Means)**: Standardized features (`YearsOfExperience`, `TeacherRating`, `AvgCourseRating`, `TotalEnrollments`) and applied K-Means clustering to partition the 60 instructors into 4 business-aligned segments.

---

## 5. Data Description
The dataset contains four distinct relational entities:
- **Users**: 3,000 unique students with an age range of 15 to 35 years (Mean: 24.97, SD: 6.05).
- **Teachers**: 60 instructors with an age range of 27 to 50 years (Mean: 38.45, SD: 7.70). Experience ranges from 1 to 24 years (Mean: 6.28, SD: 4.72).
- **Courses**: 60 courses across 12 categories, with an average duration of 27.63 hours and price ranging from $0 (38 free courses) to $490.90 (22 paid courses).
- **Transactions**: 10,000 enrollment records spanning the calendar year 2025.

---

## 6. EDA Findings

### Teacher Demographics and Distribution
- **Expertise Areas**: Digital Marketing is the largest domain (11 teachers), followed by Cybersecurity (9) and Artificial Intelligence (6).
- **Experience**: The cohort is relatively junior, with an average experience of 6.28 years. However, a few outliers possess over 20 years of experience.
- **Teacher Rating**: The overall rating distribution is centered around a mean of 3.125, with standard deviation 0.95.

### Course Distribution
- **Category Composition**: The 60 courses are distributed perfectly evenly across 12 categories (5 courses each).
- **Course Level**: Beginner and Advanced courses represent 21 courses each, while Intermediate represents 18.
- **Course Price**: Over 63% of the courses are free, making the platform highly accessible but creating a strong dependency on paid courses for revenue.

### Enrollment Patterns (Transactions)
- A staggering **60.86%** of all 10,000 transactions are concentrated under just two instructors: **Yolanda Levine** (3,061 transactions) and **Kimberly Miller** (3,025 transactions). The remaining 58 instructors share only 39.14% of the platform's enrollments.
- These two instructors teach 53 and 55 unique courses respectively, essentially acting as the primary instructors for almost all courses on the platform.

---

## 7. Statistical Analysis & Interpretation

We calculated Pearson ($r$) and Spearman ($\rho$) correlation coefficients for seven distinct relationships at the aggregated instructor level ($n=60$):

| Relationship | Pearson ($r$) | Pearson $p$-value | Spearman ($\rho$) | Spearman $p$-value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YearsOfExperience vs TeacherRating** | 0.5980 | 0.000000 | 0.6705 | 0.000000 | **Yes** |
| **YearsOfExperience vs AvgCourseRating** | -0.0562 | 0.669492 | -0.0441 | 0.737670 | No |
| **TeacherRating vs AvgCourseRating** | 0.0003 | 0.998333 | -0.0545 | 0.679325 | No |
| **TeacherRating vs TotalEnrollments** | 0.3183 | 0.013197 | -0.0120 | 0.927360 | **Yes** (Pearson only) |
| **AvgCourseRating vs TotalEnrollments** | -0.0386 | 0.769798 | -0.1780 | 0.173731 | No |
| **TeacherAge vs TeacherRating** | 0.3520 | 0.005817 | 0.3333 | 0.009260 | **Yes** |
| **TeacherAge vs AvgCourseRating** | -0.0255 | 0.846505 | -0.0509 | 0.699567 | No |

### Interpretations:
1. **Experience vs Teacher Rating**: A strong, positive correlation ($r = 0.598, p < 0.001$) shows that student satisfaction with the instructor is heavily driven by teaching experience.
2. **Experience vs Course Rating**: There is no statistical relationship ($p = 0.669$). Experienced teachers do not necessarily associate with higher-rated course materials.
3. **Teacher Rating vs Course Rating**: Pearson correlation is close to zero ($r = 0.0003, p = 0.998$). This is a critical finding: **instructor delivery and course content are completely decoupled**. A stellar teacher can lead a poorly structured course, and a weak teacher can teach an excellently designed course.
4. **Teacher Rating vs Enrollments**: The Pearson coefficient is moderately positive and significant ($r = 0.318, p = 0.013$), but the Spearman coefficient is non-significant ($\rho = -0.0120, p = 0.927$). This mathematical discrepancy indicates that the linear correlation is driven by the extreme outliers (Yolanda Levine and Kimberly Miller), who have both high ratings and massive enrollments. For the general cohort, rating does not drive enrollment.
5. **Age vs Ratings**: Teacher age is moderately correlated with teacher rating ($r = 0.352, p = 0.005$) but has no impact on course rating ($p = 0.846$), mirroring the experience trends.

---

## 8. KPI Analysis

The platform-wide KPIs reveal structural vulnerabilities:
1. **Average Teacher Rating (3.125 / 5.00)**: Moderate, indicating room for instructional quality improvement.
2. **Average Course Rating (3.098 / 5.00)**: Low to moderate, showing a clear need for curricular review.
3. **Rating Consistency Index (43.39 / 100)**: Low. High variation in ratings across courses taught by the same teachers suggests lack of content standardization.
4. **Instructor Reliability Index (27.14 / 100)**: Very low, reflecting high rating volatility and a large portion of low-rated, inexperienced instructors.
5. **Enrollment Influence Ratio (1.05)**: Suggests that high-rated courses ($>4.0$) receive only 5% more enrollments on average than low-rated courses ($\le 3.0$). Enrollments are driven by factors other than course rating (such as teacher association).
6. **Course Excellence Index (26.67%)**: Only 26.67% of courses on the platform are rated "High" ($>4.0$), representing a major quality risk.
7. **Best Expertise Domain**: **Marketing** holds the highest average course rating of **3.65**.

---

## 9. Advanced Analytics: K-Means Segmentation
The K-Means clustering algorithm identified 4 distinct segments among the 60 instructors:

1. **Star Performers ($n=2$, 3.3% of cohort)**:
   - *Profile*: Highly experienced veterans (Mean: 22.5 years experience, 47.5 years old). Exceptional teacher ratings (Mean: 4.78).
   - *Operational Impact*: Handle **6,086 enrollments (60.86% of total)**. They teach across 50+ courses each.
2. **Growing Instructors ($n=15$, 25.0% of cohort)**:
   - *Profile*: Mid-career teachers (Mean: 7.53 years experience). Good ratings (Mean: 3.74), but relatively low average course ratings (Mean: 2.83) and low enrollments (Mean: 70.2).
   - *Operational Impact*: Act as a strong middle tier with high potential but currently low leverage.
3. **Average Instructors ($n=20$, 33.3% of cohort)**:
   - *Profile*: Experienced teachers (Mean: 7.9 years experience) with moderate ratings (Mean: 3.61) and solid course ratings (Mean: 3.39), but low enrollment (Mean: 57.75).
   - *Operational Impact*: Consistent performers who are underutilized.
4. **At-Risk Instructors ($n=23$, 38.3% of cohort)**:
   - *Profile*: Novice teachers (Mean: 2.65 years experience, 33.5 years old) with poor ratings (Mean: 2.16) and low enrollments (Mean: 74.2).
   - *Operational Impact*: Represent a major quality hazard; they constitute the largest segment of the workforce (38.3%).

---

## 10. Business Insights & Recommendations

### Key Insights:
- **The "Two-Teacher Dependancy" Risk**: If Yolanda Levine or Kimberly Miller leaves the platform, EduPro risk losing 60% of its active student base. This concentration of human capital is highly dangerous.
- **Novice Quality Hazard**: 38.3% of the instructors are classified as "At-Risk" due to low experience and poor ratings. They need immediate training or replacement.
- **Content-Teacher Disconnect**: The total decoupling of teacher and course ratings means that content quality must be managed separately from instructor quality. Excellent teaching cannot save poor slides/exercises.

### Action Plan:
1. **De-risk Enrollment Concentration**: Introduce a load-balancing system where high-potential "Growing" and "Average" teachers are assigned to co-teach or lead sections of the popular courses currently dominated by the Star Performers.
2. **Mentorship Programs**: Pair the "Star Performers" with the 23 "At-Risk Instructors" in a formal mentorship program to transfer pedagogy skills.
3. **Curricular Standardization**: Perform a comprehensive review of the 73.33% of courses that fall below a 4.0 rating, standardizing syllabus designs and assets across all categories.
4. **Incentive Tiers**: Tie compensation to a composite performance index (incorporating teacher rating, course rating, and student consistency) rather than experience alone, encouraging continuous improvement.

---

## 11. Conclusion
By integrating transactional, demographic, and academic data, this study has exposed deep structural challenges on the EduPro platform, most notably the extreme concentration of enrollments in two instructors and a high proportion of low-rated, inexperienced staff. However, these challenges represent significant opportunities for strategic optimization. By scaling growing instructors, separating content quality management from delivery, and mitigating human capital risks, EduPro can successfully drive both educational excellence and commercial viability.
