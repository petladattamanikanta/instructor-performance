import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Page Config and Setup
st.set_page_config(
    page_title="EduPro Evaluation Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #312e81;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Navigation Sidebar Customization */
    .css-1d391kg {
        background-color: #0f172a !important;
    }
    
    /* Section dividers */
    .section-header {
        border-bottom: 2px solid #312e81;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #6366f1;
        font-weight: 700;
    }
    
    /* Info Box */
    .info-card {
        background-color: #1e1b4b;
        border-left: 5px solid #6366f1;
        padding: 15px;
        border-radius: 6px;
        color: #e2e8f0;
        margin-bottom: 20px;
    }
    
    /* Recommendation Card */
    .rec-card {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Data Loading & Joining
# -------------------------------------------------------------
@st.cache_data
def load_data():
    # Verify file existence and load
    data_dir = "data"
    
    master_df = pd.read_csv(os.path.join(data_dir, "master_analytical_data.csv"))
    leaderboard_df = pd.read_csv(os.path.join(data_dir, "teacher_leaderboard.csv"))
    correlations_df = pd.read_csv(os.path.join(data_dir, "correlations.csv"))
    expertise_df = pd.read_csv(os.path.join(data_dir, "expertise_summary.csv"))
    category_df = pd.read_csv(os.path.join(data_dir, "course_category_summary.csv"))
    level_df = pd.read_csv(os.path.join(data_dir, "course_level_summary.csv"))
    kpis_df = pd.read_csv(os.path.join(data_dir, "kpi_metrics.csv"))
    
    # Process dates
    master_df['TransactionDate'] = pd.to_datetime(master_df['TransactionDate'])
    master_df['Month'] = master_df['TransactionDate'].dt.strftime('%Y-%m')
    master_df['MonthNum'] = master_df['TransactionDate'].dt.month
    
    # Merge additional instructor metrics back to master df for dynamic filters
    cols_to_merge = ['TeacherID', 'ConsistencyScore', 'ReliabilityScore', 'PerformanceScore', 'InstructorSegment']
    # Drop columns if they already exist in master_df to avoid duplicate suffixing
    cols_existing = [c for c in cols_to_merge if c in master_df.columns and c != 'TeacherID']
    if cols_existing:
        master_df = master_df.drop(columns=cols_existing)
        
    master_df = master_df.merge(leaderboard_df[cols_to_merge], on='TeacherID', how='left')
    
    return master_df, leaderboard_df, correlations_df, expertise_df, category_df, level_df, kpis_df

try:
    master_df, leaderboard_df, correlations_df, expertise_df, category_df, level_df, kpis_df = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}. Please ensure data preprocessing has completed successfully.")
    st.stop()

# -------------------------------------------------------------
# 3. Sidebar Filters
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/graduation-cap.png", width=70)
st.sidebar.title("EduPro Analytics Portal")
st.sidebar.write("Explore instructor and course quality evaluation.")

st.sidebar.markdown('<div class="section-header">Page Navigation</div>', unsafe_allow_html=True)
pages = [
    "Overview & Platform Health",
    "Instructor Leaderboard & Profile",
    "Course Quality & Metrics",
    "Experience Impact Analysis",
    "Expertise Domain Performance",
    "Advanced Machine Learning",
    "Recommendations & Data Download"
]
selected_page = st.sidebar.radio("Select Page", pages)

st.sidebar.markdown('<div class="section-header">Global Interactive Filters</div>', unsafe_allow_html=True)

# 1. Teacher Selector
teachers_list = sorted(list(master_df['TeacherName'].dropna().unique()))
selected_teachers = st.sidebar.multiselect("Select Instructors", teachers_list, default=[])

# 2. Expertise Selector
expertise_list = sorted(list(master_df['Expertise'].dropna().unique()))
selected_expertise = st.sidebar.multiselect("Select Expertise Areas", expertise_list, default=[])

# 3. Gender Selector
gender_list = sorted(list(master_df['TeacherGender'].dropna().unique()))
selected_gender = st.sidebar.multiselect("Select Teacher Gender", gender_list, default=[])

# 4. Course Category Selector
category_list = sorted(list(master_df['CourseCategory'].dropna().unique()))
selected_category = st.sidebar.multiselect("Select Course Categories", category_list, default=[])

# 5. Course Level Selector
level_list = sorted(list(master_df['CourseLevel'].dropna().unique()))
selected_level = st.sidebar.multiselect("Select Course Levels", level_list, default=[])

# 6. Rating Range Sliders
min_tr_rating = float(master_df['TeacherRating'].min())
max_tr_rating = float(master_df['TeacherRating'].max())
selected_teacher_rating_range = st.sidebar.slider(
    "Teacher Rating Range",
    min_tr_rating, max_tr_rating, (min_tr_rating, max_tr_rating)
)

min_cr_rating = float(master_df['CourseRating'].min())
max_cr_rating = float(master_df['CourseRating'].max())
selected_course_rating_range = st.sidebar.slider(
    "Course Rating Range",
    min_cr_rating, max_cr_rating, (min_cr_rating, max_cr_rating)
)

# 7. Experience Range Slider
min_exp = int(master_df['YearsOfExperience'].min())
max_exp = int(master_df['YearsOfExperience'].max())
selected_exp_range = st.sidebar.slider(
    "Years of Experience Range",
    min_exp, max_exp, (min_exp, max_exp)
)

# -------------------------------------------------------------
# 4. Filtering Logic
# -------------------------------------------------------------
def apply_filters(df):
    filtered = df.copy()
    if selected_teachers:
        filtered = filtered[filtered['TeacherName'].isin(selected_teachers)]
    if selected_expertise:
        filtered = filtered[filtered['Expertise'].isin(selected_expertise)]
    if selected_gender:
        filtered = filtered[filtered['TeacherGender'].isin(selected_gender)]
    if selected_category:
        filtered = filtered[filtered['CourseCategory'].isin(selected_category)]
    if selected_level:
        filtered = filtered[filtered['CourseLevel'].isin(selected_level)]
        
    filtered = filtered[
        (filtered['TeacherRating'] >= selected_teacher_rating_range[0]) & 
        (filtered['TeacherRating'] <= selected_teacher_rating_range[1])
    ]
    filtered = filtered[
        (filtered['CourseRating'] >= selected_course_rating_range[0]) & 
        (filtered['CourseRating'] <= selected_course_rating_range[1])
    ]
    filtered = filtered[
        (filtered['YearsOfExperience'] >= selected_exp_range[0]) & 
        (filtered['YearsOfExperience'] <= selected_exp_range[1])
    ]
    return filtered

filtered_master = apply_filters(master_df)

# Filtered teacher-level leaderboard data
filtered_teachers_leaderboard = leaderboard_df[
    leaderboard_df['TeacherID'].isin(filtered_master['TeacherID'])
].copy()

# Recalculate dynamic statistics for filtered master dataset if filters are applied
if filtered_master.empty:
    st.warning("⚠️ No records match the selected filters. Please expand your criteria in the sidebar.")
    st.stop()

# -------------------------------------------------------------
# PAGE 1: Overview & Platform Health
# -------------------------------------------------------------
if selected_page == "Overview & Platform Health":
    st.title("🎓 Executive Overview & Platform Health")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">High-level business intelligence metrics and transactional trends for EduPro.</p>', unsafe_allow_html=True)
    
    # Grid of KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    # Calculate filtered KPIs
    avg_teacher_r = filtered_master['TeacherRating'].mean()
    avg_course_r = filtered_master['CourseRating'].mean()
    enrollment_count = filtered_master['TransactionID'].nunique()
    total_revenue = filtered_master['Amount'].sum()
    
    # Course Excellence (Percent of courses rated >4)
    unique_courses = filtered_master.drop_duplicates(subset=['CourseID'])
    course_exc = (unique_courses['CourseRating'] > 4.0).mean() * 100 if not unique_courses.empty else 0
    
    # Teacher Consistency (Average Consistency Score of active teachers)
    unique_teachers = filtered_teachers_leaderboard.drop_duplicates(subset=['TeacherID'])
    consistency_idx = unique_teachers['ConsistencyScore'].mean() if not unique_teachers.empty else 0
    reliability_idx = unique_teachers['ReliabilityScore'].mean() if not unique_teachers.empty else 0
    
    kpi_col1.metric("Avg Teacher Rating", f"{avg_teacher_r:.2f} ★")
    kpi_col2.metric("Avg Course Rating", f"{avg_course_r:.2f} ★")
    kpi_col3.metric("Total Enrollments", f"{enrollment_count:,}")
    kpi_col4.metric("Total Platform Revenue", f"${total_revenue:,.2f}")
    
    st.markdown("---")
    
    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
    kpi_col5.metric("Course Excellence Index", f"{course_exc:.1f}%", help="Percent of courses rated > 4.0")
    kpi_col6.metric("Rating Consistency Score", f"{consistency_idx:.1f}/100", help="Measures consistency of ratings across taught courses")
    kpi_col7.metric("Instructor Reliability Index", f"{reliability_idx:.1f}/100", help="Weighted index of rating and consistency")
    
    # Experience Impact (Pearson coefficient on active subset)
    if len(unique_teachers) > 1:
        exp_impact_val, _ = pearsonr(unique_teachers['YearsOfExperience'], unique_teachers['TeacherRating'])
    else:
        exp_impact_val = 0
    kpi_col8.metric("Experience Impact Coef", f"{exp_impact_val:.3f}", help="Pearson correlation between years of experience and teacher rating")
    
    st.markdown('<div class="section-header">Core Enrollment & Financial Trends</div>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📅 Monthly Enrollment Trends (2025)")
        # Group by Month
        monthly_df = filtered_master.groupby('Month').agg(
            Enrollments=('TransactionID', 'count'),
            Revenue=('Amount', 'sum')
        ).reset_index()
        
        # Sort months correctly
        monthly_df = monthly_df.sort_values(by='Month')
        
        fig_monthly = px.line(
            monthly_df, x='Month', y='Enrollments',
            title='Monthly Platform Enrollments',
            labels={'Enrollments': 'Total Enrollments', 'Month': 'Billing Month'},
            markers=True,
            color_discrete_sequence=['#6366f1']
        )
        fig_monthly.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_monthly, use_container_width=True)
        
    with chart_col2:
        st.subheader("📚 Category Popularity & Volume")
        cat_counts = filtered_master.groupby('CourseCategory').size().reset_index(name='EnrollmentCount')
        cat_counts = cat_counts.sort_values(by='EnrollmentCount', ascending=True)
        
        fig_cat = px.bar(
            cat_counts, y='CourseCategory', x='EnrollmentCount',
            orientation='h',
            title='Enrollments by Course Category',
            labels={'CourseCategory': 'Category', 'EnrollmentCount': 'Enrollments'},
            color='EnrollmentCount',
            color_continuous_scale=px.colors.sequential.Purples
        )
        fig_cat.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown('<div class="section-header">Rating Distribution Overview</div>', unsafe_allow_html=True)
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.subheader("📊 Instructor Rating Distribution")
        fig_tr_dist = px.histogram(
            unique_teachers, x='TeacherRating',
            nbins=15,
            title='Distribution of Instructor Ratings',
            labels={'TeacherRating': 'Rating'},
            color_discrete_sequence=['#818cf8'],
            marginal='box'
        )
        fig_tr_dist.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_tr_dist, use_container_width=True)
        
    with dist_col2:
        st.subheader("📊 Course Rating Distribution")
        fig_cr_dist = px.histogram(
            unique_courses, x='CourseRating',
            nbins=15,
            title='Distribution of Course Ratings',
            labels={'CourseRating': 'Rating'},
            color_discrete_sequence=['#34d399'],
            marginal='box'
        )
        fig_cr_dist.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cr_dist, use_container_width=True)

# -------------------------------------------------------------
# PAGE 2: Instructor Leaderboard & Profile
# -------------------------------------------------------------
elif selected_page == "Instructor Leaderboard & Profile":
    st.title("👨‍🏫 Instructor Performance & Profile Finder")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Identify top and bottom performers, and look up detailed individual scorecards.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Instructor Cohort Leaderboards</div>', unsafe_allow_html=True)
    
    # Performance score sorting
    sorted_teachers = filtered_teachers_leaderboard.sort_values(by='PerformanceScore', ascending=False)
    
    tab_top, tab_bottom = st.tabs(["Top 10 Instructors", "Bottom 10 Instructors"])
    
    cols_to_show = [
        'TeacherID', 'TeacherName', 'Expertise', 'YearsOfExperience', 
        'TeacherRating', 'AvgCourseRating', 'TotalEnrollments', 
        'ConsistencyScore', 'ReliabilityScore', 'PerformanceScore', 'InstructorSegment'
    ]
    
    with tab_top:
        st.write("🌟 Top 10 instructors based on weighted Performance Score:")
        st.dataframe(sorted_teachers[cols_to_show].head(10), use_container_width=True)
        
    with tab_bottom:
        st.write("⚠️ Bottom 10 instructors based on weighted Performance Score:")
        st.dataframe(sorted_teachers[cols_to_show].tail(10), use_container_width=True)
        
    st.markdown('<div class="section-header">Enrollment Concentration Scatter Analysis</div>', unsafe_allow_html=True)
    
    # Scatter plot: Rating vs Enrollments
    fig_scatter = px.scatter(
        filtered_teachers_leaderboard,
        x='TeacherRating', y='TotalEnrollments',
        size='YearsOfExperience', color='InstructorSegment',
        hover_name='TeacherName',
        title='Teacher Rating vs. Total Enrollments (Bubble Size = Years of Experience)',
        labels={'TeacherRating': 'Instructor Rating', 'TotalEnrollments': 'Total Enrollments', 'InstructorSegment': 'Segment'},
        color_discrete_map={
            "Star Performers": "#f43f5e",
            "Growing Instructors": "#10b981",
            "Average Instructors": "#3b82f6",
            "At-Risk Instructors": "#f59e0b"
        }
    )
    fig_scatter.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown('<div class="section-header">🔎 Individual Profile Lookup</div>', unsafe_allow_html=True)
    
    selected_profile_name = st.selectbox("Select an Instructor to view profile:", sorted(list(filtered_teachers_leaderboard['TeacherName'].unique())))
    
    if selected_profile_name:
        profile_row = filtered_teachers_leaderboard[filtered_teachers_leaderboard['TeacherName'] == selected_profile_name].iloc[0]
        
        prof_col1, prof_col2 = st.columns([1, 2])
        
        with prof_col1:
            st.markdown(f"### Scorecard: {selected_profile_name}")
            st.write(f"**ID:** {profile_row['TeacherID']}")
            st.write(f"**Expertise Area:** {profile_row['Expertise']}")
            st.write(f"**Experience:** {profile_row['YearsOfExperience']} Years ({profile_row['ExperienceTier']})")
            st.write(f"**Teacher Rating:** {profile_row['TeacherRating']} / 5.0 ({profile_row['TeacherRatingTier']})")
            st.write(f"**Avg Course Rating:** {profile_row['AvgCourseRating']} / 5.0")
            st.write(f"**Total Enrollments:** {profile_row['TotalEnrollments']}")
            st.write(f"**Segment Class:** {profile_row['InstructorSegment']}")
            
            # Gauge for Performance Score
            fig_gauge = px_go.Figure(px_go.Indicator(
                mode = "gauge+number",
                value = profile_row['PerformanceScore'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Performance Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#6366f1"},
                    'steps' : [
                        {'range': [0, 40], 'color': "#475569"},
                        {'range': [40, 75], 'color': "#1e293b"},
                        {'range': [75, 100], 'color': "#312e81"}
                    ],
                }
            ))
            fig_gauge.update_layout(template='plotly_dark', height=250, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with prof_col2:
            st.markdown("### Courses Taught & Volume")
            # Get courses and transaction counts for this teacher
            teacher_courses = filtered_master[filtered_master['TeacherName'] == selected_profile_name]
            course_summary = teacher_courses.groupby(['CourseID', 'CourseName', 'CourseCategory', 'CourseLevel', 'CourseRating']).size().reset_index(name='StudentCount')
            
            st.dataframe(course_summary.sort_values(by='StudentCount', ascending=False), use_container_width=True)
            
            # Course chart
            fig_t_courses = px.bar(
                course_summary, x='CourseName', y='StudentCount',
                color='CourseRating',
                title='Enrollments per Course',
                labels={'CourseName': 'Course', 'StudentCount': 'Students'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_t_courses.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_t_courses, use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: Course Quality & Metrics
# -------------------------------------------------------------
elif selected_page == "Course Quality & Metrics":
    st.title("📚 Course Quality & Content Evaluation")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Analyze how course ratings, categories, and levels impact platform outcomes.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Course Category Deep Dive</div>', unsafe_allow_html=True)
    
    course_cat = filtered_master.groupby('CourseCategory').agg(
        TotalEnrollments=('TransactionID', 'count'),
        AvgCourseRating=('CourseRating', 'mean'),
        CourseCount=('CourseID', 'nunique')
    ).reset_index()
    
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        st.subheader("⭐ Average Course Rating by Category")
        fig_cat_rating = px.bar(
            course_cat.sort_values(by='AvgCourseRating', ascending=True),
            x='AvgCourseRating', y='CourseCategory',
            orientation='h',
            title='Avg Course Rating by Category',
            color='AvgCourseRating',
            color_continuous_scale=px.colors.sequential.Agsunset
        )
        fig_cat_rating.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cat_rating, use_container_width=True)
        
    with cat_col2:
        st.subheader("📦 Treemap: Enrollment Volume & Ratings")
        fig_tree = px.treemap(
            filtered_master, path=['CourseCategory', 'CourseName'],
            values='Amount',
            color='CourseRating',
            title='Course Sales Matrix (Color = Rating, Box Size = Sales Revenue)',
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig_tree.update_layout(template='plotly_dark')
        st.plotly_chart(fig_tree, use_container_width=True)
        
    st.markdown('<div class="section-header">Course Difficulty & Structure Analysis</div>', unsafe_allow_html=True)
    
    lvl_col1, lvl_col2 = st.columns(2)
    
    with lvl_col1:
        st.subheader("📶 Ratings by Course Level (Beginner vs. Advanced)")
        # Box plot showing levels vs ratings
        unique_c_lvl = filtered_master.drop_duplicates(subset=['CourseID'])
        fig_lvl_box = px.box(
            unique_c_lvl, x='CourseLevel', y='CourseRating',
            color='CourseLevel',
            points='all',
            title='Course Ratings by Level',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_lvl_box.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lvl_box, use_container_width=True)
        
    with lvl_col2:
        st.subheader("👥 Enrollments by Course Level")
        lvl_enroll = filtered_master.groupby('CourseLevel').size().reset_index(name='Enrollments')
        fig_lvl_pie = px.pie(
            lvl_enroll, values='Enrollments', names='CourseLevel',
            hole=0.4,
            title='Enrollment Share by Course Difficulty',
            color_discrete_sequence=['#4338ca', '#3b82f6', '#10b981']
        )
        fig_lvl_pie.update_layout(template='plotly_dark')
        st.plotly_chart(fig_lvl_pie, use_container_width=True)

    st.markdown('<div class="section-header">Matrix Heatmap: Category vs. Level Rating</div>', unsafe_allow_html=True)
    
    # Heatmap logic
    pivot_df = unique_c_lvl.pivot_table(
        index='CourseCategory', columns='CourseLevel', 
        values='CourseRating', aggfunc='mean'
    ).fillna(0)
    
    fig_heat = px.imshow(
        pivot_df,
        labels=dict(x="Difficulty Level", y="Course Category", color="Avg Rating"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='Viridis',
        title='Average Course Rating by Category & Level'
    )
    fig_heat.update_layout(template='plotly_dark')
    st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------------------------------------
# PAGE 4: Experience Impact Analysis
# -------------------------------------------------------------
elif selected_page == "Experience Impact Analysis":
    st.title("📈 Impact of Experience on Ratings & Outcomes")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Investigate if teaching experience drives student ratings and enrollment success.</p>', unsafe_allow_html=True)
    
    exp_col1, exp_col2 = st.columns(2)
    
    unique_t_exp = filtered_teachers_leaderboard.drop_duplicates(subset=['TeacherID'])
    
    with exp_col1:
        st.subheader("🔍 Experience vs. Teacher Rating")
        fig_exp_tr = px.scatter(
            unique_t_exp, x='YearsOfExperience', y='TeacherRating',
            trendline='ols',
            color='ExperienceTier',
            hover_name='TeacherName',
            title='Does Experience Improve Ratings? (OL S Regression)',
            labels={'YearsOfExperience': 'Years of Experience', 'TeacherRating': 'Teacher Rating'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_exp_tr.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_exp_tr, use_container_width=True)
        
    with exp_col2:
        st.subheader("🔍 Experience vs. Course Rating")
        fig_exp_cr = px.scatter(
            unique_t_exp, x='YearsOfExperience', y='AvgCourseRating',
            trendline='ols',
            color='ExperienceTier',
            hover_name='TeacherName',
            title='Does Experience Improve Course Material Ratings?',
            labels={'YearsOfExperience': 'Years of Experience', 'AvgCourseRating': 'Avg Course Rating'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_exp_cr.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_exp_cr, use_container_width=True)
        
    st.markdown('<div class="section-header">Performance Across Experience Tiers</div>', unsafe_allow_html=True)
    
    tier_stats = unique_t_exp.groupby('ExperienceTier').agg(
        TeacherCount=('TeacherID', 'count'),
        AvgTeacherRating=('TeacherRating', 'mean'),
        AvgCourseRating=('AvgCourseRating', 'mean'),
        AvgEnrollment=('TotalEnrollments', 'mean')
    ).reset_index()
    
    t_col1, t_col2, t_col3 = st.columns(3)
    
    with t_col1:
        st.subheader("👥 Teacher Counts")
        fig_t_cnt = px.bar(
            tier_stats, x='ExperienceTier', y='TeacherCount',
            title='Number of Instructors by Experience Tier',
            color_discrete_sequence=['#4f46e5']
        )
        fig_t_cnt.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t_cnt, use_container_width=True)
        
    with t_col2:
        st.subheader("⭐ Teacher & Course Ratings")
        # Melt to plot both in grouped bar chart
        melted_tiers = tier_stats.melt(id_vars='ExperienceTier', value_vars=['AvgTeacherRating', 'AvgCourseRating'], 
                                       var_name='Metric', value_name='Rating')
        fig_t_ratings = px.bar(
            melted_tiers, x='ExperienceTier', y='Rating', color='Metric',
            barmode='group',
            title='Ratings by Experience Tier',
            color_discrete_sequence=['#6366f1', '#10b981']
        )
        fig_t_ratings.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t_ratings, use_container_width=True)
        
    with t_col3:
        st.subheader("📈 Average Student Enrollments")
        fig_t_enroll = px.bar(
            tier_stats, x='ExperienceTier', y='AvgEnrollment',
            title='Avg Enrollments per Instructor by Tier',
            color_discrete_sequence=['#fbbf24']
        )
        fig_t_enroll.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t_enroll, use_container_width=True)

# -------------------------------------------------------------
# PAGE 5: Expertise Domain Performance
# -------------------------------------------------------------
elif selected_page == "Expertise Domain Performance":
    st.title("🎯 Expertise Area Analytics")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Compare outcomes across academic expertise areas to identify key domain drivers.</p>', unsafe_allow_html=True)
    
    unique_teachers = filtered_teachers_leaderboard.drop_duplicates(subset=['TeacherID'])
    
    # Calculate stats per expertise
    exp_summary = unique_teachers.groupby('Expertise').agg(
        TeacherCount=('TeacherID', 'count'),
        AvgTeacherRating=('TeacherRating', 'mean'),
        AvgCourseRating=('AvgCourseRating', 'mean'),
        TotalEnrollments=('TotalEnrollments', 'sum')
    ).reset_index()
    
    st.markdown('<div class="section-header">Expertise Overview Matrix</div>', unsafe_allow_html=True)
    st.dataframe(exp_summary.sort_values(by='TotalEnrollments', ascending=False), use_container_width=True)
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.subheader("🎯 Average Ratings by Domain")
        melted_exp = exp_summary.melt(id_vars='Expertise', value_vars=['AvgTeacherRating', 'AvgCourseRating'],
                                      var_name='Metric', value_name='Rating')
        fig_exp_bar = px.bar(
            melted_exp, x='Expertise', y='Rating', color='Metric',
            barmode='group',
            title='Teacher vs Course Rating by Domain',
            color_discrete_sequence=['#4f46e5', '#10b981']
        )
        fig_exp_bar.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_exp_bar, use_container_width=True)
        
    with exp_col2:
        st.subheader("👥 Total Transaction Volume by Expertise")
        fig_exp_pie = px.pie(
            exp_summary, values='TotalEnrollments', names='Expertise',
            hole=0.4,
            title='Enrollment Share by Domain Expertise',
            color_discrete_sequence=px.colors.qualitative.D3
        )
        fig_exp_pie.update_layout(template='plotly_dark')
        st.plotly_chart(fig_exp_pie, use_container_width=True)

# -------------------------------------------------------------
# PAGE 6: Advanced Machine Learning
# -------------------------------------------------------------
elif selected_page == "Advanced Machine Learning":
    st.title("🤖 Advanced Analytics: Segmentation & Correlations")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Investigate instructor cohorts using K-Means Clustering and evaluate linear/rank statistical relationships.</p>', unsafe_allow_html=True)
    
    tab_clust, tab_corr = st.tabs(["K-Means Instructor Segmentation", "Correlation Heatmap & Hypothesis Testing"])
    
    with tab_clust:
        st.markdown('<div class="section-header">K-Means Clustering Profile</div>', unsafe_allow_html=True)
        
        clust_features = ['YearsOfExperience', 'TeacherRating', 'AvgCourseRating', 'TotalEnrollments']
        
        # Display centroid metrics
        clust_summary = filtered_teachers_leaderboard.groupby('InstructorSegment')[clust_features + ['TeacherID']].agg(
            AvgExperience=('YearsOfExperience', 'mean'),
            AvgTeacherRating=('TeacherRating', 'mean'),
            AvgCourseRating=('AvgCourseRating', 'mean'),
            AvgEnrollments=('TotalEnrollments', 'mean'),
            InstructorCount=('TeacherID', 'count')
        ).reset_index()
        
        st.write("📈 Dynamic Cluster Summary Table:")
        st.dataframe(clust_summary, use_container_width=True)
        
        # 3D Scatter plot
        st.subheader("3D Cluster Interactive Plot")
        fig_3d = px.scatter_3d(
            filtered_teachers_leaderboard,
            x='YearsOfExperience', y='TeacherRating', z='TotalEnrollments',
            color='InstructorSegment',
            hover_name='TeacherName',
            title='3D Segmentation Matrix (Experience, Teacher Rating, Total Enrollments)',
            color_discrete_map={
                "Star Performers": "#f43f5e",
                "Growing Instructors": "#10b981",
                "Average Instructors": "#3b82f6",
                "At-Risk Instructors": "#f59e0b"
            }
        )
        fig_3d.update_layout(template='plotly_dark', height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("""
        ### Cluster Interpretation Guidelines:
        * **Star Performers**: Veteran teachers with decades of experience, near-perfect student ratings, and dominant enrollment shares.
        * **Growing Instructors**: Mid-career instructors showing good student response but limited course offerings or student share.
        * **Average Instructors**: Dependable performers with average experience and standard course feedback.
        * **At-Risk Instructors**: Junior instructors with poor feedback and minimal student retention. Needs targeted support.
        """)
        
    with tab_corr:
        st.markdown('<div class="section-header">Statistical Correlation Details</div>', unsafe_allow_html=True)
        st.write("Below are the Pearson ($r$) and Spearman ($\rho$) correlation results along with statistical significance tests ($p$-values):")
        st.dataframe(correlations_df, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Correlation Matrix Visualization")
        # Build correlation matrix of numerical columns for teachers
        teacher_num = filtered_teachers_leaderboard[['Age', 'YearsOfExperience', 'TeacherRating', 'AvgCourseRating', 'TotalEnrollments', 'PerformanceScore']].corr()
        
        fig_corr_mat = px.imshow(
            teacher_num,
            text_auto=".3f",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='Pearson Correlation Matrix (Instructor Aggregates)'
        )
        fig_corr_mat.update_layout(template='plotly_dark')
        st.plotly_chart(fig_corr_mat, use_container_width=True)
        
        st.markdown("""
        **Critical Hypotheses Interpreted:**
        1. **Years of Experience vs. Teacher Rating**: *Statistically Significant* ($p < 0.001$). Experience is a powerful predictor of instructor delivery success.
        2. **Teacher Rating vs. Course Rating**: *Not Significant* ($p = 0.998$). Student perception of teaching does not correlate with their evaluation of the course contents. Quality audits must treat content and instruction as separate issues.
        """)

# -------------------------------------------------------------
# PAGE 7: Recommendations & Data Download
# -------------------------------------------------------------
elif selected_page == "Recommendations & Data Download":
    st.title("📋 Strategic Recommendations & Reports")
    st.markdown('<p style="font-size:1.1rem; color:#64748b;">Actionable strategies for EduPro stakeholders and download reports.</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Strategic Action Plan (Stakeholder Brief)</div>', unsafe_allow_html=True)
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("""
        <div class="rec-card">
            <h4>🚨 High Priority: De-risk the 'Two-Teacher Dependency'</h4>
            <p><strong>Finding:</strong> Two instructors (Yolanda Levine and Kimberly Miller) teach almost every course and hold over 60% of all platform enrollments.</p>
            <p><strong>Recommendation:</strong> Introduce section leaders and assign 'Growing' or 'Average' instructors as co-teachers in highly populated courses. This distributes student load and creates backup capacity.</p>
        </div>
        <div class="rec-card">
            <h4>📊 High Priority: Curricular Standardization</h4>
            <p><strong>Finding:</strong> Only 26.67% of courses on the platform are rated high (> 4.0), and course rating has no statistical link to instructor rating.</p>
            <p><strong>Recommendation:</strong> Standardize course slides, quizzes, and projects separate from teachers. Target underperforming domains (like Machine Learning and Web Development courses) for slide audits.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rec2:
        st.markdown("""
        <div class="rec-card">
            <h4>💡 Medium Priority: Mentorship Circles</h4>
            <p><strong>Finding:</strong> 38.3% of the instructor pool consists of 'At-Risk' instructors with less than 3 years of experience and ratings under 2.2.</p>
            <p><strong>Recommendation:</strong> Task the two 'Star Performers' to lead mandatory teaching workshops. Set up a review system where novice instructors' recorded sessions are peer-reviewed.</p>
        </div>
        <div class="rec-card">
            <h4>⚙️ Medium Priority: Performance-Tied Incentives</h4>
            <p><strong>Finding:</strong> Experience strongly determines rating, indicating teachers develop skills over time.</p>
            <p><strong>Recommendation:</strong> Restructure payment tiers. Provide bonuses for teachers who maintain a TeacherRating above 4.0 and show consistency across multiple courses, incentivizing professional growth.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📥 Report Downloads & Data Export</div>', unsafe_allow_html=True)
    
    st.write("Download preprocessed CSV datasets for external reporting or auditing:")
    
    d_col1, d_col2, d_col3 = st.columns(3)
    
    # 1. Master analytical data download
    csv_master = filtered_master.to_csv(index=False).encode('utf-8')
    d_col1.download_button(
        label="Download Filtered Master Dataset (CSV)",
        data=csv_master,
        file_name="edupro_filtered_master_dataset.csv",
        mime="text/csv"
    )
    
    # 2. Leaderboard download
    csv_lead = filtered_teachers_leaderboard.to_csv(index=False).encode('utf-8')
    d_col2.download_button(
        label="Download Instructor Leaderboard (CSV)",
        data=csv_lead,
        file_name="edupro_instructor_leaderboard.csv",
        mime="text/csv"
    )
    
    # 3. Correlations download
    csv_corr = correlations_df.to_csv(index=False).encode('utf-8')
    d_col3.download_button(
        label="Download Correlation Statistics (CSV)",
        data=csv_corr,
        file_name="edupro_correlation_stats.csv",
        mime="text/csv"
    )
    
    st.success("Report export links generated. Click any button above to download raw files.")
