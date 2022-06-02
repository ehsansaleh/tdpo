#include "defs.hpp"   // all definitions

/////////////////////////////////////////////////////
//////////////// Utility Functions //////////////////
/////////////////////////////////////////////////////
void solve_sec_ode(double a, double b, double c,
                   double y_init, double ydot_init,
                   double t_f, double* y_tf, double* ydot_tf);

void solve_motor_ode(double tau_cmd, double y_init, double ydot_init,
                     double t_f, double* y_tf, double* ydot_tf);

inline void _get_contact_state(mjModel* mj_model, mjData* mj_data,
  int ground_geom_id, int lowerleg_limb_geom_id,
  bool* foot_in_contact, double foot_force[6]);

/////////////////////////////////////////////////////
////////// RewardGiver Class Definitions ////////////
/////////////////////////////////////////////////////

class SimInterface;

class SimplifiedRewardGiver {
  public:
    SimplifiedRewardGiver();                         // This is the constructor
    //~SimplifiedRewardGiver();                      // This is the destructor
    void reset(mjModel* mj_model,
               mjData* mj_data,
               SimInterface* simintf);               // re-initialize the reward giver state

    void update_reward(double obs_current[obs_dim],  // This should be current, since we'll be applying a
                                                     // reward delay (i.e., the same as observation delay)
                       mjModel* mj_model,
                       mjData* mj_data,
                       double* reward);

    void update_unstable_reward(double* reward);      // The reward for encountering simulation instability

  private:
    bool do_compute_mjids;
    bool has_touched_ground;
    int foot_center_site_mjid;
    int hip_center_site_mjid;
    int knee_center_site_mjid;
    int ground_geom_id;
    int lowerleg_limb_geom_id;
    double foot_force[6];
    SimInterface* simif;

    #if add_act_sm_r == True
      int hip_ctrl_id, knee_ctrl_id;
      double traj_tau_hip[done_inner_steps];
      double traj_tau_knee[done_inner_steps];
    #endif

    #ifdef Debug_reward
      int no_update_calls;
      int slider_mjid, hip_mjid, knee_mjid;
    #endif
};

class OriginalRewardGiver {
  public:
    OriginalRewardGiver();                           // This is the constructor
    //~OriginalRewardGiver();                        // This is the destructor
    void reset(mjModel* mj_model,
               mjData* mj_data,
               SimInterface* simintf);               // re-initialize the reward giver state

    void update_reward(double obs_current[obs_dim],  // This should be current, since we'll be applying a
                                                     // reward delay (i.e., the same as observation delay)
                       mjModel* mj_model,
                       mjData* mj_data,
                       double* reward);

  void update_unstable_reward(double* reward);      // The reward for encountering simulation instability

  private:
    bool do_compute_mjids;
    bool has_touched_ground;
    int foot_center_site_mjid;
    int hip_center_site_mjid;
    int knee_center_site_mjid;
    int ground_geom_id;
    int lowerleg_limb_geom_id;
    int base_body_mjid;
    int upperleg_body_mjid;
    int lowerleg_body_mjid;
    int hip_ctrl_id;
    int knee_ctrl_id;
    double foot_force[6];
    double leg_vel_peak;
    double leg_mass;
    double base_body_mass_ratio;
    double upperleg_body_mass_ratio;
    double lowerleg_body_mass_ratio;
    SimInterface* simif;

    double tau_hip;
    double tau_knee;
    double tau_hip_old;
    double tau_knee_old;

    #ifdef Debug_reward
      int no_update_calls;
      int slider_mjid, hip_mjid, knee_mjid;
    #endif
};

/////////////////////////////////////////////////////
////////// SimInterface Class Definition ////////////
/////////////////////////////////////////////////////

class SimInterface {
  public:
    SimInterface();                 // This is the constructor
    ~SimInterface();                // This is the destructor
    int set_ground_contact();       // This will set the ground contact solimp and solref
                                    // to be either compliant or noncompliant.

    double* reset(double theta_hip, double theta_knee,
                  double omega_hip, double omega_knee,
                  double pos_slider, double vel_slider,
                  double jump_time_in,
                  int noise_index);     // resets the robot to a particular state.

    double* add_trq_buff(double new_trqs[]);
    double* add_obs_buff(double new_obs[]);
    double  add_rew_buff(double new_rew);
    void update_mj_obs();
    void step_inner(double action_raw[]);
    void step(double action_raw[act_dim],
              double** next_obs,
              double* reward,
              bool* done);

    // step counts
    int inner_step_count;
    int outer_step_count;

    int hip_ctrl_id;  // Used for setting the hip actuation
    int knee_ctrl_id; // Used for setting the knee actuation
    int jump_state;
  private:
    // mujoco elements
    mjModel* mj_model;
    mjData* mj_data;
    int slider_mjid;
    int hip_mjid;
    int knee_mjid;

    // motor model items
    double motor_tau_state_hip;
    double motor_tau_state_hip_dot;
    double motor_tau_state_knee;
    double motor_tau_state_knee_dot;

    // The raw observations from mujoco (i.e., non-delayed)
    double mj_obs[obs_dim];
    // #define theta_hip mj_obs[0]
    // #define theta_knee mj_obs[1]
    // #define omega_hip mj_obs[2]
    // #define omega_knee mj_obs[3]

    // observation delay buffer items
    double obs_delay_buff[obs_dly_buflen][obs_dim];
    bool obs_dlybuf_ever_pushed;
    int obs_dlybuf_push_idx;
    // #define theta_hip_dlyd  obs_delay_buff[obs_dlybuf_push_idx][0]
    // #define theta_knee_dlyd obs_delay_buff[obs_dlybuf_push_idx][1]
    // #define omega_hip_dlyd  obs_delay_buff[obs_dlybuf_push_idx][2]
    // #define omega_knee_dlyd obs_delay_buff[obs_dlybuf_push_idx][3]

    // torque delay buffer items
    double trq_delay_buff[trq_dly_buflen][act_dim];
    bool trq_dlybuf_ever_pushed;
    int trq_dlybuf_push_idx;
    // #define tau_hip_dlyd  trq_delay_buff[trq_dlybuf_push_idx][0]
    // #define tau_knee_dlyd trq_delay_buff[trq_dlybuf_push_idx][1]

    // reward delay buffer items
    double rew_delay_buff[rew_dly_buflen];
    bool rew_dlybuf_ever_pushed;
    int rew_dlybuf_push_idx;

    // The final torque values that will be sent to mujoco's controls
    double joint_torque_hip_capped;
    double joint_torque_knee_capped;

    bool is_mj_stable;

    // The reward giver object
    RewardGiver rew_giver;

    #if action_type == torque
      // No joint_torque_command variable creation is necessary
    #elif action_type == jointspace_pd
      double joint_torque_command[act_dim];
    #elif action_type == workspace_pd
      #error "you possibly need some joint_torque_command definition here."
    #else
      #error "action_type not implemented."
    #endif

    #if do_jump == True
      double jump_time;
    #endif

    #if do_obs_noise == True
      double non_noisy_omega_hip;
      double non_noisy_omega_knee;
      int noise_idx;
      double noise_arr[noise_rows][2] = {{-0.092803872965597, 1.0791227818568663},
        {-0.15820480948543925, 0.9951208935687395},
        {-0.2104163884743162, 0.8873352397750907},
        {-0.2784625859624942, 0.798052215437907},
        {-0.33384637276894336, 0.6967386456218772},
        {-0.4111741766771495, 0.593853252799029},
        {-0.466284949894443, 0.4656002420913721},
        {-0.5346660006077402, 0.39661735448886315},
        {-0.548188232226122, 0.31603514823267354},
        {-0.6218642229868052, 0.24048415955139113},
        {-0.6483670994864035, 0.19755688401363125},
        {-0.6917205862159079, 0.17702802535664253},
        {-0.7201454557353038, 0.10962664505409414},
        {-0.761418733026094, 0.021123365361550572},
        {-0.766714299957912, -0.08959156483636788},
        {-0.8107337380497341, -0.14510350935360838},
        {-0.7908659946464152, -0.20176223952376482},
        {-0.7984599495922651, -0.23461743944399194},
        {-0.7632802183603278, -0.1961682302735528},
        {-0.7606168067918335, -0.1764160774593102},
        {-0.724145617325193, -0.09887198274962339},
        {-0.6752770145856792, -0.05569965828438139},
        {-0.6851642862573544, 0.07128738113964328},
        {-0.6605853335160203, 0.12770681654187488},
        {-0.6141877540155372, 0.22627168194662328},
        {-0.550140011909036, 0.294315360817889},
        {-0.5331894918343867, 0.3837196363029629},
        {-0.4896625562293395, 0.4591763677709806},
        {-0.45009325357312546, 0.59298734737864},
        {-0.4118664844747939, 0.7107320228277474},
        {-0.3739939145212241, 0.8506963112116783},
        {-0.3398286225772196, 0.9310681417121307},
        {-0.28759306808054275, 1.0040928839708703},
        {-0.24270281444598618, 1.0445455340860623},
        {-0.21152587698709668, 1.1418634591960424},
        {-0.16724535189219725, 1.1863438000584066},
        {-0.15000998972010526, 1.2219498817868644},
        {-0.12514710833603493, 1.2357717880692252},
        {-0.08970182373426239, 1.2727110391708476},
        {-0.09660367476321619, 1.2778044850168087},
        {-0.06823007526482794, 1.2346433342666572},
        {-0.06893757525685795, 1.1874120864940565},
        {-0.04828409845331083, 1.1062242542408067},
        {-0.0816260407606273, 1.0225827166470016},
        {-0.06950180161695751, 0.907412031873216},
        {-0.1013533749582134, 0.8087339309433124},
        {-0.12841405698057917, 0.583032971293767},
        {-0.1892318922914602, 0.32656188363822025},
        {-0.18434649097824596, -0.0426702953625524},
        {-0.17584532190558577, -0.2529189780091987},
        {-0.21833188407822712, -0.4710718736440267},
        {-0.2681769696755554, -0.572920430748808},
        {-0.2972767249654753, -0.6298075299100803},
        {-0.32986803113953966, -0.6089247368928499},
        {-0.32668072190232955, -0.4594905699203795},
        {-0.35535068295987804, -0.3433754702463818},
        {-0.3632102791732701, -0.204968755169324},
        {-0.3713994543176191, -0.15032408587273105},
        {-0.3617309322805897, -0.10883988273114342},
        {-0.3303952516843931, -0.09713073809645234},
        {-0.3394847736456663, -0.0495583102630448},
        {-0.32211829630841704, -0.018907571667991174},
        {-0.2785257354577235, 0.0590697372220097},
        {-0.21810127955147007, 0.12451114319045642},
        {-0.17525540253628358, 0.2532169963714699},
        {-0.10624968891941755, 0.34716224686584285},
        {-0.052356231688595045, 0.513080519820873},
        {0.0009095350436103544, 0.5957309701559819},
        {0.025604545556489722, 0.7499295175920895},
        {0.042583627327126505, 0.8023014654479792},
        {0.09453237790651947, 0.8989189246645317},
        {0.15334953894531478, 0.9705922977387234},
        {0.18814070795147186, 1.0773708571173377},
        {0.20967577286117733, 1.1604144959648108},
        {0.22543811053124974, 1.2486183858250168},
        {0.22554969828369975, 1.2664274061413794},
        {0.20646223726937496, 1.309321713204004},
        {0.21310008710855066, 1.2847916778584478},
        {0.19645913578402463, 1.282020026528027},
        {0.1433146547759132, 1.26860972286521},
        {0.11755633812659028, 1.119034504628142},
        {0.11393588235562424, 1.0099253564519914},
        {0.09472125945369791, 0.9090351945760746},
        {0.10522602200386144, 0.7454528828854325},
        {0.08529209112813652, 0.49223537245193416},
        {-0.019740887598155066, 0.031912200337980146},
        {-0.15271634503304288, -0.2267649790599391},
        {-0.21135031014657368, -0.45662977751758627},
        {-0.2956276806840421, -0.5210765682770857},
        {-0.3622985317081624, -0.5484560262586777},
        {-0.4067039265011143, -0.48306788080775},
        {-0.4737947985270248, -0.16847211660237704},
        {-0.5519876938076345, 0.07394776055503183},
        {-0.5680548502243807, 0.3638448336429567},
        {-0.6228948933929961, 0.45928273994509716},
        {-0.6739523168512558, 0.505483247474614},
        {-0.713972430234759, 0.5165530947248165},
        {-0.7017843314458463, 0.5451292001719503},
        {-0.72061835997502, 0.5537369798870113},
        {-0.6960406883727233, 0.564577504626695},
        {-0.6299812707704193, 0.5927500679742215},
        {-0.5849318522653961, 0.6437760976280575},
        {-0.5475909776925203, 0.6977134910067813},
        {-0.5291910613521726, 0.7041984771374432},
        {-0.499292173789057, 0.6780140320711707},
        {-0.4740844713896246, 0.6408010637863288},
        {-0.43028473057958694, 0.5876599345096754},
        {-0.39485333697133207, 0.4741502144040819},
        {-0.3306861298737265, 0.3443293749097478},
        {-0.32148156246495097, -0.043007417847064744},
        {-0.3002548883508802, -0.3114726581633187},
        {-0.2873970069097078, -0.588611636492824},
        {-0.24557331390214632, -0.6568370813902131},
        {-0.21687910357552465, -0.39674161962472176},
        {-0.19024910563820452, -0.12347047673270417},
        {-0.1774712525519031, 0.22963109483532307},
        {-0.12095422625484531, 0.37388528284938616},
        {-0.09158532026763178, 0.4688327276144264},
        {-0.08447394238511796, 0.47160180198179624},
        {-0.058541260234273196, 0.5214105916156582},
        {-0.006467978788610829, 0.5803085848839995},
        {0.01845940412726943, 0.6145792860753376},
        {0.05741188721701285, 0.6429640374647043},
        {0.1146605975987427, 0.7213395785371324},
        {0.13501244454702865, 0.7609176138482763},
        {0.14178146907382727, 0.812167957552238},
        {0.08613959779209956, 0.8332860227079895},
        {0.056049655781812646, 0.8771367578428579},
        {0.0015053957176061061, 0.9053020978492663},
        {-0.00827903457024437, 0.858683049200839},
        {-0.021602546014358293, 0.8148284118604661},
        {-0.0375439045493291, 0.7480645419063334},
        {-0.10410576151825612, 0.7169638291826477},
        {-0.13007258135980893, 0.6396117836241029},
        {-0.15708214888895133, 0.5498963731291449},
        {-0.1509509671983449, 0.4364613661750303},
        {-0.1509825827211002, 0.3349978930350579},
        {-0.1817692154038899, -0.07010203108162649},
        {-0.18079053218678354, -0.35048928283875114},
        {-0.18078536474560147, -0.647772319013356},
        {-0.1760693036175658, -0.6301310869982313},
        {-0.20128352384227366, -0.27323508913063055},
        {-0.2224096363167969, -0.002089071208724036},
        {-0.25961036554689176, 0.373027213298049},
        {-0.3060230125675045, 0.530726681235496},
        {-0.3206005391165112, 0.7056153337760875},
        {-0.3272500433150851, 0.7870428503682971},
        {-0.34974446831513495, 0.8738352211206957},
        {-0.36401596232063227, 0.9290883669565888},
        {-0.3892390917332067, 0.9644560465151581},
        {-0.406666819822628, 0.9982582573463761},
        {-0.41159937346416564, 1.0358784129794785},
        {-0.3849302160514747, 1.0436555465149473},
        {-0.3478371610732176, 1.0024539157395855},
        {-0.3518717416448651, 0.9203748204712436},
        {-0.3296252553230894, 0.7871327783645983},
        {-0.29744590355989065, 0.6807447250663365},
        {-0.28114471632853055, 0.4718758625444943},
        {-0.3162125920439989, 0.37158878470276857},
        {-0.3076269894261974, 0.19351819943738846},
        {-0.32624630393891074, 0.04936608413131438},
        {-0.31822198368237586, -0.13107041916992568},
        {-0.2886108902249944, -0.08068545486440915},
        {-0.26935972064492475, 0.14549376917191914},
        {-0.2518310894480087, 0.2539213049157647},
        {-0.24472648907910965, 0.3655329824716409},
        {-0.240558777861704, 0.3767286815186752},
        {-0.21181329503082003, 0.3868708280015145},
        {-0.18879367886743426, 0.3648639874789925},
        {-0.15488331427640967, 0.34485474575363373},
        {-0.12880058448212672, 0.3393201287372811},
        {-0.11632361962695237, 0.3072879406133229},
        {-0.09453446240428764, 0.31100615938562726},
        {-0.07712272499441086, 0.33734602434055816},
        {-0.08040731801170198, 0.33958040938372314},
        {-0.05480855213401048, 0.3515662841297287},
        {0.011001189128243105, 0.3619717762632426},
        {0.07223404540577505, 0.35917285216513584},
        {0.11956945264968555, 0.35630331594150544},
        {0.10454421119564694, 0.3248636595289094},
        {0.08418801112192975, 0.3292079975049358},
        {0.050459106298412326, 0.290074103481893},
        {0.038125854563229034, 0.23376197371255714},
        {0.0396057165328787, -0.04786850131909404},
        {0.002364790287776586, -0.23310631432113738},
        {-0.0313721985415687, -0.3385016140210748},
        {-0.06988603729809251, -0.294965348972406},
        {-0.09383467050198213, -0.12620679243027766},
        {-0.14685480949222907, 0.0057689694567883976},
        {-0.17299959566169365, 0.22021140069902856},
        {-0.23484556309847893, 0.3657885654142179},
        {-0.2450160991251189, 0.5458962936911842},
        {-0.2749635908148642, 0.6603521593510333},
        {-0.296683348211376, 0.8234237115478908},
        {-0.3366075694279771, 0.9210715626933643},
        {-0.34954804039785303, 1.0289801566829082},
        {-0.3497961738386639, 1.0600089777345048},
        {-0.335510383632053, 1.1523226451642463},
        {-0.328002779574303, 1.2061698985048537},
        {-0.2918753522369637, 1.2224689973323182},
        {-0.27812028917061715, 1.1875613347865768},
        {-0.28742419488932747, 1.1079486871803041},
        {-0.29697165116413027, 1.0350185040193347},
        {-0.2644680534941135, 0.8846474213977249},
        {-0.28976537339375863, 0.7369844491187161},
        {-0.28627737756686367, 0.4926734347026729},
        {-0.26884353136694905, 0.3199577599248986},
        {-0.2618940243516765, 0.053087726597712326},
        {-0.2529500916685894, -0.13158212441029615},
        {-0.25304677422321964, -0.21097156799960448},
        {-0.22491316689397456, -0.07065517516504993},
        {-0.2475835687705863, 0.13590178882285908},
        {-0.23286306873050622, 0.2288672543484127},
        {-0.21526122284223215, 0.2909164234418782},
        {-0.22774104776702142, 0.30405257837848154},
        {-0.23205076114281997, 0.2720851391200334},
        {-0.21601098067636304, 0.2737850756767477},
        {-0.2042461239012363, 0.24896400224280235},
        {-0.1662200359927537, 0.23111158350590255},
        {-0.1420923032452941, 0.22482297193325973},
        {-0.10335624801087473, 0.20540631923709007},
        {-0.08078104929856034, 0.20428549421341025},
        {-0.02760942609864081, 0.2123977186479964},
        {0.010287862043083962, 0.22396384792011492},
        {0.004311933141179836, 0.1896272630924205},
        {-0.008598167137944035, 0.11887918982147383},
        {-0.019216193711872442, 0.09150380348486742},
        {-0.02829832307312108, 0.04761876227370809},
        {-0.016931792507385968, 0.07698706480647388},
        {0.028146543551230696, 0.047580446883321414},
        {0.08838875465365659, -0.04736527067980223},
        {0.1114871871161176, -0.2973207435389318},
        {0.098950258591044, -0.392465188639858},
        {0.07753880502591359, -0.3949510414887545},
        {0.008447750353208772, -0.31319160002590696},
        {0.00022496670732596868, -0.1875008867352177},
        {-0.04747402464547079, -0.06384915061378926},
        {-0.08920647338096277, 0.0918510003376305},
        {-0.11653177351969735, 0.21743548914599575},
        {-0.13063219100172674, 0.3561870278037098},
        {-0.15770749568641795, 0.4382967447360002},
        {-0.16078673684410782, 0.5391203698842171},
        {-0.17234178316575965, 0.5958453223359887},
        {-0.16902622016410218, 0.6439361254635436},
        {-0.17638989263417448, 0.6863443193545393},
        {-0.1990580681294425, 0.7147973203614404},
        {-0.1809115294931778, 0.7496232496051807},
        {-0.15679069464292272, 0.7337845675198782},
        {-0.13612472638418538, 0.6886594620698414},
        {-0.1683901496825495, 0.5775060920644197},
        {-0.1936954357663616, 0.39392538599878346},
        {-0.19791981930389557, -0.04208352431096074},
        {-0.22683835638595395, -0.3408724498389706},
        {-0.24698930473052938, -0.6851672537577702},
        {-0.2661535802659962, -0.8260820593238565},
        {-0.2742199069422391, -0.8103095058655878},
        {-0.3269067238207901, -0.5935184381955336},
        {-0.32643167747878676, -0.21193030745675312},
        {-0.3090148407422988, 0.013269886947649745},
        {-0.2587605800039934, 0.19709682346698187},
        {-0.24846316266780066, 0.29213897090688423},
        {-0.23566936060359822, 0.3992615010805389},
        {-0.2100032013755695, 0.4428159970017598},
        {-0.16988672982677988, 0.4965835172197428},
        {-0.09220878630192608, 0.5527185546999256},
        {-0.04266539557442872, 0.6414554735847169},
        {0.02982865960169656, 0.7081269890773516},
        {0.061312818465979024, 0.8129491289353403},
        {0.12648293686298473, 0.8690377291581841},
        {0.15041342320215412, 0.9312254513308016},
        {0.15858137839503694, 0.9501912287238552},
        {0.1430713084723303, 0.9408451630861165},
        {0.09029406485261582, 0.8783207732880305},
        {0.08394464257997791, 0.8063626379572266},
        {0.03638921379135196, 0.7817388457826224},
        {0.003444583395572298, 0.7083436723444549},
        {-0.02775879916627888, 0.6516786439761288},
        {-0.08165243628683028, 0.55211804530287},
        {-0.13334812259208872, 0.4558222279440507},
        {-0.19880284819864036, 0.3107829903533261},
        {-0.258120181418128, 0.19855885287157093},
        {-0.313565605035941, -0.10342987042889273},
        {-0.37282401705521506, -0.41190168930070126},
        {-0.3964936416760869, -0.7978751535707627},
        {-0.4522479778247841, -0.8982072697875623},
        {-0.5314823092190579, -0.97586771785625},
        {-0.5825119061075217, -0.9847229625106433},
        {-0.5978833407308115, -0.7908716142666314},
        {-0.6501323473874496, -0.5177732536563093},
        {-0.6568733918744214, -0.0759520365943418},
        {-0.6787199704394773, 0.12538127118118947},
        {-0.6651077761198358, 0.3389847789679923},
        {-0.6591151181492152, 0.37721385816174546},
        {-0.6487522940112345, 0.4665938932971141},
        {-0.6146847328109168, 0.5270085383600334},
        {-0.5643616925095571, 0.5541574957756019},
        {-0.46202593312491214, 0.6304293069517701},
        {-0.3955403944607965, 0.6937390887703554},
        {-0.286071537375733, 0.7529912226272035},
        {-0.1997161648038337, 0.768117985204336},
        {-0.0859709794748329, 0.7749069027027904},
        {-0.039398473426459635, 0.7267593727159056},
        {0.03843783443109272, 0.7069063556526514},
        {0.08407107376275835, 0.595790603128064},
        {0.10363552917584684, 0.49092368525559227},
        {0.13476451609575824, 0.11455784730707208},
        {0.14212658336315043, -0.18811878535358506},
        {0.17261013462280772, -0.5430235021641474},
        {0.1848433392020854, -0.6638700746613244},
        {0.18089204276959858, -0.5154581404088256},
        {0.16745807329687645, -0.2276504803880428},
        {0.1584282586178838, 0.14376718214279105},
        {0.11321621590780007, 0.30003088450393367},
        {0.09535826524782287, 0.45091943765042775},
        {0.030779488491070595, 0.4973005497685996},
        {0.010003238248519786, 0.5397480363350722},
        {-0.033262683263327375, 0.567190673024502},
        {-0.024614655142187303, 0.6382765439683098},
        {-0.04577697750337917, 0.6595202591291818},
        {-0.032884712774959635, 0.7215190918235459},
        {-0.007315765902750737, 0.7392938542947727},
        {-9.882607389144482e-05, 0.7595712137071322},
        {0.05043166406921529, 0.7918037874648558},
        {0.040944258024051106, 0.7882164646388752},
        {0.06009507692330862, 0.7796753796439395},
        {0.06188428826846604, 0.7282257352167276},
        {0.06965794650812152, 0.7133558836741729},
        {0.05754114473333649, 0.6514425162255817},
        {-0.03987033447881849, 0.6260141964219601},
        {-0.11241573788281967, 0.5402105368584271},
        {-0.16938780004148457, 0.4940222657030642},
        {-0.19269550469319574, 0.39859613027290397},
        {-0.20472942845306052, 0.36212015351707816},
        {-0.19915766554116798, 0.26491746961090357},
        {-0.24459945678481443, 0.21462551250107786},
        {-0.2940306469473626, 0.1024141349612826},
        {-0.3798551222333799, -0.031017928689291097},
        {-0.4108315346704403, -0.39042820779953047},
        {-0.466581219601526, -0.5816897408368367},
        {-0.4843576931630764, -0.7433313253012339},
        {-0.4911853031200315, -0.7566008502222488},
        {-0.49550170038484165, -0.5164468198269141},
        {-0.5462292232352537, -0.2944028888857013},
        {-0.5493970875770624, 0.03715128675485957},
        {-0.5372147548541806, 0.19582518726020925},
        {-0.534111686868957, 0.3576102949378184},
        {-0.4913250491354413, 0.4161611736583062},
        {-0.49124256308822867, 0.5149594945706335},
        {-0.44238560697306983, 0.5726101650582041},
        {-0.42152410444211164, 0.6478211888028746},
        {-0.2967613099691979, 0.734844869717028},
        {-0.24819434816242403, 0.861192630448175},
        {-0.1564374794261063, 0.900678685576414},
        {-0.10931902520305758, 0.9707213583052514},
        {-0.04182706629138089, 1.0162285008266503},
        {-0.005313858971626839, 1.0568353440294889},
        {0.07366956633269495, 1.0541242204901153},
        {0.09803067132653553, 0.9840310182440786},
        {0.13409666939758091, 0.9154119330235919},
        {0.14847673180162713, 0.8122649054767974},
        {0.17896572622485052, 0.7357181772795833},
        {0.17224065254826493, 0.6293279150937208},
        {0.201224258586562, 0.5573319030107888},
        {0.18944795584438046, 0.38761373222366036},
        {0.19391169085120885, 0.3126117308651377},
        {0.19074491121330595, 0.1689675809640332},
        {0.14290898901020466, 0.1014942503641203},
        {0.13699301380171103, -0.05591195560682083},
        {0.08667629688128509, -0.11855130749342457},
        {0.06337318766555855, -0.16030794209799692},
        {0.026597076826769817, -0.10261356716383618},
        {-0.007921354595050056, 0.03759038514298263},
        {-0.027672967140950577, 0.08913639219617187},
        {-0.07117309026123775, 0.06863775673951977},
        {-0.07774066311040273, 0.08556124405279686},
        {-0.08283269581133257, 0.06629392342633356},
        {-0.10286278759258827, 0.05516243259561948},
        {-0.13625457660729157, 0.025009686637251338},
        {-0.17614493404853593, 0.013131781803543241},
        {-0.1982059998754686, 0.022723976606327856},
        {-0.21593181344933465, 0.023641774366874202},
        {-0.22721995675363038, 0.03049832548429654},
        {-0.23659811321806168, 0.03754489712233644},
        {-0.2048641121727086, 0.01230985651341543},
        {-0.21167181100495647, 0.006899019113016269},
        {-0.20843065953959883, 0.005123064709946057},
        {-0.216643971477551, 0.019917070592702313},
        {-0.20393333321206586, 0.03941902211574},
        {-0.16393419607407367, 0.06138374748746278},
        {-0.1545957992518825, 0.0843036272267077},
        {-0.12581441537172688, 0.10608146928962858},
        {-0.09610407457838988, 0.07999885504468329},
        {-0.049277909569306555, 0.10446260321037082},
        {-0.03596413629634543, 0.14505724242656637},
        {-0.030715299701145282, 0.13652062234369744},
        {-0.01304433136662908, 0.16533801268015935},
        {-0.004558341126567278, 0.20199531826764483},
        {-0.0008855104336509267, 0.19507309087243518},
        {0.028113638742391922, 0.22718119678720772},
        {0.02931505357567321, 0.24614820614133315},
        {0.02931505357567321, 0.24614820614133315}};
        // The last row was duplicated due to MeasuredNoiseGenerator's inner workings.
    #endif
};