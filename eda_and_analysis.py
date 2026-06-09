import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_analysis():
    print("Starting Statistical Analysis and Modeling Phase...")
    
    # Load files
    master_df = pd.read_csv("data/master_analytical_data.csv")
    teachers = pd.read_csv("data/cleaned_teachers.csv")
    courses = pd.read_csv("data/cleaned_courses.csv")
    transactions = pd.read_csv("data/cleaned_transactions.csv")
    users = pd.read_csv("data/cleaned_users.csv")
    
    # -------------------------------------------------------------
    # 1. Instructor Aggregation for Leaderboard
    # -------------------------------------------------------------
    print("Aggregating instructor-level statistics...")
    
    # For each teacher, get:
    # - Average course rating of courses they are associated with in transactions
    # - Standard deviation of course ratings (for consistency)
    # - Unique courses they teach
    # - Total transactions (enrollments)
    
    # Let's map teachers to courses they teach
    teacher_course_stats = master_df.groupby('TeacherID').agg(
        TotalEnrollments=('TransactionID', 'count'),
        AvgCourseRating=('CourseRating', 'mean'),
        StdCourseRating=('CourseRating', 'std'),
        UniqueCoursesTaught=('CourseID', 'nunique')
    ).reset_index()
    
    # Merge with teacher dimensions
    teacher_leaderboard = teachers.merge(teacher_course_stats, on='TeacherID', how='left').fillna({
        'TotalEnrollments': 0,
        'AvgCourseRating': 0.0,
        'StdCourseRating': 0.0,
        'UniqueCoursesTaught': 0
    })
    
    # Calculate Consistency Score: 100 * (1 - StdCourseRating / 4.0), max std on 1-5 scale is ~2.0
    # Let's define it as: Consistency = 100 * (1 - StdCourseRating / 2.0). Bounded between 0 and 100.
    # If Std is NaN or 0 (only 1 course taught), we default it to 90.0 (high consistency but not perfect due to lack of variance data)
    def calc_consistency(row):
        std = row['StdCourseRating']
        if pd.isna(std) or std == 0:
            if row['UniqueCoursesTaught'] > 1:
                return 100.0
            return 90.0
        score = 100.0 * (1.0 - (std / 2.0))
        return max(0.0, min(100.0, score))
        
    teacher_leaderboard['ConsistencyScore'] = teacher_leaderboard.apply(calc_consistency, axis=1)
    
    # Reliability Score: Combined rating and consistency
    # Reliability = (TeacherRating / 5.0) * ConsistencyScore
    teacher_leaderboard['ReliabilityScore'] = (teacher_leaderboard['TeacherRating'] / 5.0) * teacher_leaderboard['ConsistencyScore']
    
    # Performance Score: Combined rating, course rating, enrollments, experience
    # Scale variables first to 0-100
    def min_max_scale(series):
        if series.max() == series.min():
            return series * 0.0 + 50.0
        return 100.0 * (series - series.min()) / (series.max() - series.min())
        
    norm_tr = min_max_scale(teacher_leaderboard['TeacherRating'])
    norm_cr = min_max_scale(teacher_leaderboard['AvgCourseRating'])
    norm_en = min_max_scale(teacher_leaderboard['TotalEnrollments'])
    norm_ex = min_max_scale(teacher_leaderboard['YearsOfExperience'])
    
    teacher_leaderboard['PerformanceScore'] = (norm_tr * 0.35) + (norm_cr * 0.35) + (norm_en * 0.20) + (norm_ex * 0.10)
    
    # Round metrics
    teacher_leaderboard['ConsistencyScore'] = teacher_leaderboard['ConsistencyScore'].round(2)
    teacher_leaderboard['ReliabilityScore'] = teacher_leaderboard['ReliabilityScore'].round(2)
    teacher_leaderboard['PerformanceScore'] = teacher_leaderboard['PerformanceScore'].round(2)
    teacher_leaderboard['AvgCourseRating'] = teacher_leaderboard['AvgCourseRating'].round(2)
    
    # Save Leaderboard
    teacher_leaderboard.to_csv("data/teacher_leaderboard.csv", index=False)
    print("Teacher Leaderboard created.")

    # -------------------------------------------------------------
    # 2. Correlation Analysis
    # -------------------------------------------------------------
    print("Calculating correlations and p-values...")
    
    # Define relationships to test:
    # 1. YearsOfExperience vs TeacherRating
    # 2. YearsOfExperience vs CourseRating
    # 3. TeacherRating vs CourseRating
    # 4. TeacherRating vs Enrollment
    # 5. CourseRating vs Enrollment
    # 6. Age vs TeacherRating
    # 7. Age vs CourseRating
    
    relationships = [
        ("YearsOfExperience vs TeacherRating", teacher_leaderboard['YearsOfExperience'], teacher_leaderboard['TeacherRating']),
        ("YearsOfExperience vs AvgCourseRating", teacher_leaderboard['YearsOfExperience'], teacher_leaderboard['AvgCourseRating']),
        ("TeacherRating vs AvgCourseRating", teacher_leaderboard['TeacherRating'], teacher_leaderboard['AvgCourseRating']),
        ("TeacherRating vs TotalEnrollments", teacher_leaderboard['TeacherRating'], teacher_leaderboard['TotalEnrollments']),
        ("AvgCourseRating vs TotalEnrollments", teacher_leaderboard['AvgCourseRating'], teacher_leaderboard['TotalEnrollments']),
        ("TeacherAge vs TeacherRating", teacher_leaderboard['Age'], teacher_leaderboard['TeacherRating']),
        ("TeacherAge vs AvgCourseRating", teacher_leaderboard['Age'], teacher_leaderboard['AvgCourseRating'])
    ]
    
    corr_results = []
    for desc, x, y in relationships:
        # Pearson
        p_coef, p_pval = pearsonr(x, y)
        # Spearman
        s_coef, s_pval = spearmanr(x, y)
        
        corr_results.append({
            "Relationship": desc,
            "Pearson_Coef": round(p_coef, 4),
            "Pearson_PValue": round(p_pval, 6),
            "Spearman_Coef": round(s_coef, 4),
            "Spearman_PValue": round(s_pval, 6),
            "P_Val_Significant": "Yes" if p_pval < 0.05 else "No"
        })
        
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv("data/correlations.csv", index=False)
    print("Correlation results saved.")

    # -------------------------------------------------------------
    # 3. K-Means Clustering for Instructor Segmentation
    # -------------------------------------------------------------
    print("Performing K-Means Clustering on instructors...")
    
    # Features for clustering
    cluster_features = ['YearsOfExperience', 'TeacherRating', 'AvgCourseRating', 'TotalEnrollments']
    X = teacher_leaderboard[cluster_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-Means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    teacher_leaderboard['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Interpret clusters based on centroid values and assign business names
    centroids = kmeans.cluster_centers_
    # Let's map clusters. We can characterize clusters using their standardized scores:
    # 1. Star Performers: High rating, high enrollments, high experience
    # 2. Growing Instructors: Lower/medium experience, high/medium ratings, moderate/low enrollments
    # 3. Average Instructors: Medium experience, medium ratings, medium enrollments
    # 4. At-Risk Instructors: Low ratings, low enrollments
    
    # We will identify clusters by ranking centroids
    # Let's compute average rating + average enrollment in each cluster to rank them
    cluster_stats = teacher_leaderboard.groupby('Cluster').agg(
        AvgRating=('TeacherRating', 'mean'),
        AvgCourseRating=('AvgCourseRating', 'mean'),
        AvgEnrollment=('TotalEnrollments', 'mean'),
        AvgExperience=('YearsOfExperience', 'mean'),
        Count=('TeacherID', 'count')
    ).reset_index()
    
    print("\nCluster Centroids (Original Scale):")
    print(cluster_stats)
    
    # Assign labels based on heuristic matching of cluster statistics
    # Standardize assigning:
    # Sort clusters by a performance indicator: AvgRating * AvgEnrollment
    cluster_stats['Score'] = cluster_stats['AvgRating'] * 0.4 + cluster_stats['AvgCourseRating'] * 0.4 + (cluster_stats['AvgEnrollment'] / cluster_stats['AvgEnrollment'].max()) * 0.2
    sorted_clusters = cluster_stats.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    # Map ranked positions to Segment Names:
    # Rank 0 (highest Score) -> Star Performers
    # Rank 1 -> Growing Instructors (or Average, based on experience)
    # Let's check experience for distinguishing between Growing (lower experience) and Average
    # If the second highest has lower experience, it's Growing.
    # Let's write a robust labeling logic:
    # Cluster with lowest AvgRating is At-Risk
    # Cluster with highest Score is Star Performers
    # Between the remaining two: the one with lower AvgExperience is "Growing Instructors", the other is "Average Instructors"
    
    at_risk_idx = cluster_stats.loc[cluster_stats['AvgRating'].idxmin(), 'Cluster']
    star_idx = cluster_stats.loc[cluster_stats['Score'].idxmax(), 'Cluster']
    
    remaining_idxs = [i for i in range(4) if i not in [at_risk_idx, star_idx]]
    
    if len(remaining_idxs) == 2:
        idx1, idx2 = remaining_idxs
        if cluster_stats.loc[idx1, 'AvgExperience'] < cluster_stats.loc[idx2, 'AvgExperience']:
            growing_idx = idx1
            average_idx = idx2
        else:
            growing_idx = idx2
            average_idx = idx1
    else:
        # Fallback if indices don't match
        growing_idx = remaining_idxs[0] if len(remaining_idxs) > 0 else 1
        average_idx = remaining_idxs[1] if len(remaining_idxs) > 1 else 2
        
    cluster_labels = {
        star_idx: "Star Performers",
        growing_idx: "Growing Instructors",
        average_idx: "Average Instructors",
        at_risk_idx: "At-Risk Instructors"
    }
    
    teacher_leaderboard['InstructorSegment'] = teacher_leaderboard['Cluster'].map(cluster_labels)
    teacher_leaderboard.to_csv("data/teacher_clusters.csv", index=False)
    
    # Update leaderboard in place too
    teacher_leaderboard.to_csv("data/teacher_leaderboard.csv", index=False)
    print("K-Means Clustering complete. Segments mapped and saved.")

    # -------------------------------------------------------------
    # 4. Expertise Performance Analysis
    # -------------------------------------------------------------
    print("Analyzing expertise areas...")
    expertise_stats = teacher_leaderboard.groupby('Expertise').agg(
        TeacherCount=('TeacherID', 'count'),
        AvgTeacherRating=('TeacherRating', 'mean'),
        AvgCourseRating=('AvgCourseRating', 'mean'),
        TotalEnrollments=('TotalEnrollments', 'sum')
    ).reset_index()
    
    expertise_stats['AvgTeacherRating'] = expertise_stats['AvgTeacherRating'].round(2)
    expertise_stats['AvgCourseRating'] = expertise_stats['AvgCourseRating'].round(2)
    expertise_stats.to_csv("data/expertise_summary.csv", index=False)
    print("Expertise statistics saved.")

    # -------------------------------------------------------------
    # 5. Course Quality Analysis
    # -------------------------------------------------------------
    print("Analyzing course-level statistics...")
    course_cat_stats = courses.groupby('CourseCategory').agg(
        CourseCount=('CourseID', 'count'),
        AvgCourseRating=('CourseRating', 'mean'),
        TotalEnrollments=('EnrollmentCountPerCourse', 'sum')
    ).reset_index()
    
    course_cat_stats['AvgCourseRating'] = course_cat_stats['AvgCourseRating'].round(2)
    course_cat_stats.to_csv("data/course_category_summary.csv", index=False)
    
    course_level_stats = courses.groupby('CourseLevel').agg(
        CourseCount=('CourseID', 'count'),
        AvgCourseRating=('CourseRating', 'mean'),
        TotalEnrollments=('EnrollmentCountPerCourse', 'sum')
    ).reset_index()
    
    course_level_stats['AvgCourseRating'] = course_level_stats['AvgCourseRating'].round(2)
    course_level_stats.to_csv("data/course_level_summary.csv", index=False)
    print("Course category and level statistics saved.")

    # -------------------------------------------------------------
    # 6. KPI Calculations
    # -------------------------------------------------------------
    print("Calculating overall platform KPIs...")
    avg_teacher_rating = teachers['TeacherRating'].mean()
    avg_course_rating = courses['CourseRating'].mean()
    
    # Rating Consistency Index: 100 - (Standard Deviation of teacher ratings * 20), scaled to 100.
    # Better: mean consistency score of teachers.
    rating_consistency_index = teacher_leaderboard['ConsistencyScore'].mean()
    
    # Experience Impact Score: Pearson correlation between experience and teacher rating
    experience_impact_score = corr_df.loc[corr_df['Relationship'] == "YearsOfExperience vs TeacherRating", "Pearson_Coef"].values[0]
    
    # Enrollment Influence Ratio: Average enrollment for High rated courses (>4.0) vs Low rated courses (<=3.0)
    high_rated_enroll = courses[courses['CourseRating'] > 4.0]['EnrollmentCountPerCourse'].mean()
    low_rated_enroll = courses[courses['CourseRating'] <= 3.0]['EnrollmentCountPerCourse'].mean()
    enrollment_influence_ratio = high_rated_enroll / low_rated_enroll if low_rated_enroll > 0 else 0
    
    # Instructor Reliability Index: Mean reliability score
    instructor_reliability_index = teacher_leaderboard['ReliabilityScore'].mean()
    
    # Course Excellence Index: Percentage of courses with CourseRating > 4.0
    course_excellence_index = (courses['CourseRating'] > 4.0).mean() * 100.0
    
    # Expertise Performance Score: Average course rating of the highest performing expertise domain
    best_expertise = expertise_stats.sort_values(by='AvgCourseRating', ascending=False).iloc[0]
    expertise_performance_score = best_expertise['AvgCourseRating']
    best_expertise_name = best_expertise['Expertise']
    
    kpis = {
        "KPI": [
            "Average Teacher Rating",
            "Average Course Rating",
            "Rating Consistency Index",
            "Experience Impact Score",
            "Enrollment Influence Ratio",
            "Instructor Reliability Index",
            "Course Excellence Index",
            "Expertise Performance Score (Best Domain)",
            "Best Expertise Domain Name"
        ],
        "Value": [
            round(avg_teacher_rating, 3),
            round(avg_course_rating, 3),
            round(rating_consistency_index, 2),
            round(experience_impact_score, 4),
            round(enrollment_influence_ratio, 2),
            round(instructor_reliability_index, 2),
            round(course_excellence_index, 2),
            round(expertise_performance_score, 2),
            best_expertise_name
        ]
    }
    
    kpi_df = pd.DataFrame(kpis)
    kpi_df.to_csv("data/kpi_metrics.csv", index=False)
    print("KPIs successfully calculated and saved.")
    print("All analytics completed. Done!")

if __name__ == "__main__":
    run_analysis()
