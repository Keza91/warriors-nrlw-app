import streamlit as st

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Information & App Overview", layout="wide")

# =========================
# SIDEBAR LOGO 
# =========================
with st.sidebar:
    # Place logo at the top with centered alignment
    st.markdown("<div style='text-align:center; margin-top:-10px; margin-bottom:15px;'>", unsafe_allow_html=True)
    st.logo("Wahs.png", size="large")
    st.sidebar.image("OneNZ.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
# ---------------------- STYLE ----------------------------

# THEME #
st.markdown("""
<style>
html, body, [class*="block-container"] {
    color: var(--text-color, #262c68) !important;
}
</style>
""", unsafe_allow_html=True)

# TABLE BOARDERS # 
st.markdown("""
<style>
/* --- Rounded outer edges only --- */
table {
    border-collapse: separate !important;
    border-spacing: 0;
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 12px;         /* soft corner radius */
    overflow: hidden;
}

/* Keep internal borders intact */
table td, table th {
    border: 1px solid rgba(0,0,0,0.1);
}

/* Optional: soft header background matching theme */
table thead th {
    background-color: var(--secondary-background-color, #e5e5f3);
}

/* Optional: subtle zebra striping for readability */
table tbody tr:nth-child(even) td {
    background-color: rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# TITLE #
st.markdown("""
<style>

.block-container {
    padding-top: 3rem !important;
    max-width: 1180px;
    margin: auto;
}
h1, h2, h3 {
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ---------------------------
col1, col2, col3 = st.columns([0.5, 4, 0.5])
with col1:
    st.image("Wahs.png", width=80)
with col2:
    st.markdown("<h2>Information & App Overview</h2>", unsafe_allow_html=True)
with col3:
    st.image("NRLW Logo.png", width=80)

st.markdown("---")

# ---------------------- INTRODUCTION ---------------------
st.markdown("""
### Purpose of the App

The **Warriors NRLW Prognostic Loading Tool** integrates positional data, match demands, and training structure to help
staff understand and plan **mechanical load** at both player and team level.

Each page of the app serves a different planning function, from position-specific top-ups to contextual analysis of what drives winning weeks.
Together, these tools form a cohesive workflow for **objective load management**.
""")
st.markdown("---")

# ---------------------- WHAT IS N/M ----------------------
st.markdown("### **Relative Cumulative Mechanical Load (N/m/kg)**")
st.markdown("""


**Definition:**  
Relative Cumulative Mechanical Load quantifies the average mechanical work an athlete produces per metre of movement, normalised to body mass.  
It is derived from the work-energy theorem, where mechanical work during constant acceleration is expressed as:

$$
W = F \\times s = m a \\left(\\tfrac{1}{2} a t^2\\right) = 0.5 m a^2 t^2
$$

Summing this across all acceleration and deceleration events gives total mechanical work, which is then divided by total distance (m) and by athlete mass (kg) to yield a **mass-normalised mechanical cost per metre**.


### **Equation**

$$
N/m_{rel} = \\frac{0.5 \\sum_i n_i a_i^2 t_i^2}{d}
$$

Where:  
- *nᵢ* = number of efforts in band *i*  
- *aᵢ* = representative acceleration magnitude (m·s⁻²)  
- *tᵢ* = typical duration for that band (s)  
- *d* = total distance (m)  

Deceleration bands are multiplied by **1.25** to reflect their higher eccentric mechanical cost.



### **Implementation in OpenField**

Where **N** =
            
    ((0.5 x $athlete_weight x 
    (($gen2_acceleration_band8_total_effort_count x (4.0 x 4.0) x (0.30 x 0.30)) +
    ($gen2_acceleration_band7_total_effort_count x (2.75 x 2.75) x (0.20 x 0.20)) +
    ($gen2_acceleration_band6_total_effort_count x (1.50 x 1.50) x (0.15 x 0.15)) +
    ($gen2_acceleration_band1_total_effort_count x (4.0 x 4.0) x (0.30 x 0.30) x 1.25) +
    ($gen2_acceleration_band2_total_effort_count x (2.75 x 2.75) x (0.20 x 0.20) x 1.25) +
    ($gen2_acceleration_band3_total_effort_count x (1.50 x 1.50) x (0.15 x 0.15) x 1.25))))
            
            

### **Implementation in the App**        
  In Google Sheets, we then multiply the **N** output by 100 and divide by the Total Distance
            
    100 x N / Total Distance

            
The leading `100 x`and `Total Distance` scales the values into a coach-friendly range (~1-20) without altering relationships.

---

### **Scientific Rationale**
1. **Physics foundation:**  
   Derived from the work-energy principle (Halliday et al., 2013).  
   \\( W = 0.5 m a^2 t^2 \\) represents the mechanical work for each effort.

2. **Distance normalisation:**  
   Dividing by total distance converts total mechanical work to a *rate of work per metre*, akin to a mechanical cost of transport.

3. **Mass normalisation:**  
   Cancelling mass makes the metric size-independent, allowing fair comparison across athletes of different body sizes.

4. **Banded effort structure:**  
   OpenField’s acceleration and deceleration effort counts (Bands 1-8) are treated as discrete constant-acceleration episodes with representative *aᵢ* and *tᵢ* values, aligning with device logic (Varley et al., 2012; Akenhead et al., 2013).

5. **Deceleration weighting:**  
   Decels are weighted 1.25x to reflect higher eccentric mechanical cost and braking demands (Harper & Kiely, 2018; Harper et al., 2019; Dalen et al., 2016).

6. **Interpretation:**  
   Higher N/m values indicate greater accel/decel work per metre (mechanical intensity).  
   Lower values reflect steadier locomotion and less mechanical stress.

"""

"")
st.markdown("---")

# ---------------------- WHY N/M IS BETTER ----------------
st.markdown("### Why use Newtons (N) and N/m")
st.image("Correlation Matrix.png", width=700)

st.markdown("""
            
As shown in the correlation matrix above, **Newtons** demonstrates **very strong correlations (r = 0.79-0.87)** with all acceleration and deceleration metrics.  
This confirms that Newtons accurately reflects the **mechanical demands of speed change activity**, capturing the braking, propulsive, and change-of-direction forces that dominate rugby league movement.

In contrast, traditional metrics such as **metres per minute (m/min)** are calculated by dividing total distance by session duration.  
            
While m/min provides a simple summary of **locomotor output**, it has key limitations:

- It only represents **distance and velocity**, not the underlying **mechanical effort** required to produce movement.  
- It ignores **accels, decels and collisions**, which drive the majority of mechanical stress.  
- It often **underestimates session intensity** when distance is similar but achieved through more stop-start, high-force activity.

For example, two players may each cover 5 km, but if one completes it in 50 minutes (100 m/min) and the other in 60 minutes (83 m/min), m/min suggests the second session is easier—even if it included more braking, contact, and direction changes.  
Thus, **m/min reflects volume and density, not mechanical load**.

**Acceleration Load**, as defined by Catapult, was designed to overcome this limitation by quantifying the total amount of *speed change activity*.  
It sums the **absolute values of acceleration and deceleration** sampled at 10 Hz (Duthie & Delaney, 2015), treating both directions equally.  

However, it still fails to represent mechanical load accurately because:

- It **treats braking and accelerating as identical**, despite braking being mechanically more costly.  
- It **excludes athlete mass**, ignoring individual differences in the force required to accelerate or decelerate.  
- It **adds acceleration magnitudes linearly**, not accounting for the squared relationship between acceleration and mechanical work.  
- It does not include **effort duration**, which determines total impulse and force application.

As a result, **Acceleration Load measures the frequency of movement, not the mechanical cost**, which explains its **moderate correlations (r = 0.5-0.6)** with acceleration and deceleration variables in the matrix.

By contrast, **N/m (Newtons per metre)** is derived from first principles using the **work-energy relationship (½ m a² t²)**.  
It incorporates acceleration magnitude, duration, and athlete mass to quantify the **mechanical work performed per metre of movement**. 
             
By summing all acceleration and deceleration events and normalising by total distance, N/m captures the **true mechanical cost** of locomotion, including braking and propulsive intensity.  
This is why N/m aligns so strongly with Accel/Decels and provides a more **mechanically valid and interpretable measure of external load** than either m/min or Acceleration Load.

**In summary:**  
- **m/min** measures *how far* an athlete moves per minute (locomotor output).  
- **Acceleration Load** measures *how often* the athlete changes speed (volume of activity).  
- **N/m** measures *how hard* those movements are on the body (mechanical intensity).

The high correlations between **N/m** and all Accel/Decel metrics demonstrate that it best represents the **true mechanical demands of rugby league performance**.

""")
st.markdown("---")
# ---------------------- HOW N/M DRIVES TOP-UPS ----------------
st.markdown("### How N/m is used to calculate top-ups")
st.markdown("""
The **N/m value** acts as the anchor metric for all load programming in this tool.  
It links match intensity to the mechanical work players complete in training, allowing staff to prescribe data-derived top-ups and weekly loading targets.

Because N/m is a composite outcome of mechanical events, it shows **strong correlations with accelerations, decelerations, and high-speed running distance**.
In most positions, N/m has the highest correlation coefficients with **total acceleration count** and **acceleration distance**, followed by decelerations and high-speed running (HSR).
This reflects the fact that changes in velocity and direction contribute more to mechanical intensity than steady-speed movement.

To operationalise this, regression models are built using match data where **N/m is the dependent variable** and metrics such as HSR, VHSR, Accel Efforts, and Decel Efforts are predictors.
The resulting **slopes** from these regressions quantify how much each metric contributes to overall mechanical intensity.
A larger slope indicates a stronger influence, meaning small increases in that metric drive large changes in N/m.

**Example workflow:**
1. You select a position (e.g. Centre).  
2. The app retrieves the regression slopes linking N/m to key mechanical metrics for that position.  
3. You input a target N/m, such as **8.0 N/m** to simulate match intensity.  
4. The tool then calculates how much additional HSR, VHSR, and accel/decel volume is required to achieve this target, scaled according to each metric’s slope.  
5. The output provides exact, position-specific top-up values, such as **99m HSR, 59m VHSR, 56 accelerations, 57 decelerations** etc.

This process ensures that top-ups are:
- **Objective**, based on quantified statistical relationships rather than assumptions,  
- **Position-specific**, reflecting that, for example, middles derive more N/m from deceleration density while edges derive more from HSR and VHSR,  
- **Mechanically balanced**, ensuring that load increases are spread proportionally across the metrics that drive N/m.

By planning training sessions and post-match top-ups around N/m, staff can ensure each athlete’s weekly load
closely replicates match-level mechanical intensity while maintaining an appropriate ratio between locomotive and mechanical stress.
""")
st.markdown("---")

# ---------------------- APP SECTIONS ---------------------
st.markdown("### App Structure & Usage")
st.markdown("""
| Module | Function | Typical Use |
|:--|:--|:--|
| **Positional Top-Ups** | Converts desired N/m increase (e.g. +1 N/m) into combinations of HSR, VHSR, accel/decel efforts. | Post-match or session top-ups. |
| **Team Planner** | Scales weekly training targets relative to exemplar values (P50, P75). | Designing daily team sessions. |
| **Week Planner** | Uses regression models to distribute load across D1-D5. | Building week structures that match game patterns. |
| **Season Set-Up** | Links opposition, phase, and volume modifiers for macro-cycle planning. | Long-term periodisation. |
| **Insights** | Compares contextual and winning-week differences. | Evaluating what load patterns are associated with success. |

""")
st.markdown("---")

# ---------------------- PRACTICAL EXAMPLES ---------------
st.markdown("### Practical Interpretation")
st.markdown("""
- A **centre** averaging *4.0 N/m* may reach *4.8 N/m* in high-intensity games, an increase of **20% mechanical intensity**.  
- If training only achieves 3.2 N/m, the athlete experiences a **33% shortfall** relative to competition demands.  
- By programming targeted accelerations, decelerations, and HSR top-ups, coaches can increase weekly training intensity
to match real match-play loads without unnecessary volume.

The tool allows staff to see how **each metric contributes to overall mechanical intensity**,
and to plan sessions that replicate game-specific stress without overtraining.
""")

# ---------------------- TAKEAWAY -------------------------
st.markdown("---")
st.markdown("""
### Key Takeaways
- **N/m = mechanical intensity**, a holistic measure combining locomotor and mechanical load.  
- **Highly correlated** with acceleration, deceleration, and high-speed running metrics.  
- Provides a **unified framework** linking player, team, and seasonal planning.  
- Enables **objective top-ups** and **data-driven periodisation**.
""")

st.markdown("---")
st.markdown("""
### References
- Harper, D.J., Carling, C., & Kiely, J. (2019). *High-Intensity Acceleration and Deceleration Demands in Elite Team Sports Competitive Match Play*. *Sports Medicine*. PDF: https://clok.uclan.ac.uk/id/eprint/29984/3/29984%2010.1007_s40279-019-01170-1.pdf :contentReference[oaicite:3]{index=3}  
- Dalen, T., Ingebrigtsen, J., Ettema, G., Hjelde, G.H., & Wisløff, U. (2016). *Player Load, Acceleration, and Deceleration During Forty-Five Competitive Matches of Elite Soccer*. *J Strength Cond Res, 30*(2):351–359. https://journals.lww.com/nsca-jscr/fulltext/2016/02000/player_load%2C_acceleration%2C_and_deceleration_during.8.aspx :contentReference[oaicite:4]{index=4}  
- McBurnie, A.J., Harper, D.J., et al. (2021). *Deceleration Training in Team Sports: Another Potential ‘Vaccine’ for Sports-Related Injury*. *Sports Medicine*. PDF: https://e-space.mmu.ac.uk/628647/1/McBurnie2021_Article_DecelerationTrainingInTeamSpor.pdf :contentReference[oaicite:5]{index=5}  
- Umberger, B.R., Gerritsen, K.G.M., & Martin, P.E. (2003). *A model of human muscle energy expenditure*. *Comput Methods Biomech Biomed Engin, 6*, 99–111.  
- McMahon, T.A. (1984). *Muscles, Reflexes, and Locomotion*. Princeton University Press.  
- Akenhead, R., Hayes, P.R., Thompson, K.G., & French, D. (2013). *Diminutions of acceleration and deceleration output during professional football match play.* *J Sci Med Sport, 16*, 556–561.  
- Duthie, G.M., & Delaney, J.A. (2015). *Acceleration-Based Running Intensities of Professional Rugby League Match-Play.*  

*Built for the One NZ Warriors NRLW by Kieran Griffiths, integrating applied sport science with match-based data analytics.*

**BJU399**
""")

