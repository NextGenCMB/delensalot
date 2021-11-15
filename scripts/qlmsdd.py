import sys
import params.__init__ as par

for sim_id in range(None):
    par.qlms_dd.get_sim_qlm('p_p', sim_id)
    print(sim_id)