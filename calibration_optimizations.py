import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import hippocampus_toolbox as hc_tools


def analyze_measdata_from_file(analyze_tx=[1, 2, 3, 4, 5, 6], meantype='db_mean'):
    """

    :param analyze_tx:
    :param txpos_tuning:
    :param meantype:
    :return:
    """

    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0

    measdata_filename = hc_tools.select_file()

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 0
        measured_wp_list = []

        for i, line in enumerate(measfile):

            if line == '### begin grid settings\n':
                # print('griddata found')
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                # print('Measurement data found')
                continue
            if load_description:
                # print('file description')
                print(line)

            if load_grid_settings and not load_measdata:
                # print(line)

                grid_settings = map(float, line[0:-3].split(','))
                x0 = [grid_settings[0], grid_settings[1]]
                xn = [grid_settings[2], grid_settings[3]]
                grid_dxdy = [grid_settings[4], grid_settings[5]]
                timemeas = grid_settings[6]

                data_shape_file = [int((xn[0] - x0[0]) / grid_dxdy[0] + 1), int((xn[1] - x0[1]) / grid_dxdy[1] + 1)]

                numtx = int(grid_settings[7])
                txdata = grid_settings[(2 + numtx):(2 + numtx + 3 * numtx)]

                # read tx positions
                txpos_list = []
                for itx in range(numtx):
                    itxpos = txdata[2 * itx:2 * itx + 2]
                    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                freqtx_list = []
                for itx in range(numtx):
                    freqtx_list.append(txdata[2 * numtx + itx])
                freqtx = np.asarray(freqtx_list)

                # print out
                print('filename = ' + measdata_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0] * data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdy = ' + str(grid_dxdy))
                print('tx_pos = ' + str(txpos_list))
                print('freqtx = ' + str(freqtx))

                startx = x0[0]
                endx = xn[0]
                stepx = data_shape_file[0]

                starty = x0[1]
                endy = xn[1]
                stepy = data_shape_file[1]

                xpos = np.linspace(startx, endx, stepx)
                ypos = np.linspace(starty, endy, stepy)

                wp_matx, wp_maty = np.meshgrid(xpos, ypos)

                # print(xpos)

            if load_measdata and not load_grid_settings:
                # print('read measdata')

                totnumwp += 1
                meas_data_line = map(float, line[0:-3].split(', '))
                meas_data_append_list.append(meas_data_line)

                meas_data_mat_line = np.asarray(meas_data_line)

                measured_wp_list.append(int(meas_data_mat_line[2]))
                num_tx = int(meas_data_mat_line[3])
                num_meas = int(meas_data_mat_line[4])

                first_rss = 5 + num_tx

                meas_data_mat_rss = meas_data_mat_line[first_rss:]

                rss_mat = meas_data_mat_rss.reshape([num_tx, num_meas])

                if meantype is 'lin':
                    rss_mat_lin = 10 ** (rss_mat / 10)
                    mean_lin = np.mean(rss_mat_lin, axis=1)
                    var_lin = np.var(rss_mat_lin, axis=1)
                    mean = 10 * np.log10(mean_lin)
                    var = 10 * np.log10(var_lin)
                else:
                    mean = np.mean(rss_mat, axis=1)
                    var = np.var(rss_mat, axis=1)
                    # print('var = ' + str(var))
                wp_pos = [meas_data_mat_line[0], meas_data_mat_line[1]]

                # print(str(wp_pos))
                # print(str(mean))
                # print(str(var))
                # print(str(np.concatenate((wp_pos, mean, var), axis=0)))
                plotdata_line = np.concatenate((wp_pos, mean, var), axis=0)

                plotdata_mat_lis.append(plotdata_line)
                # print (str(plotdata_mat_lis))

        measfile.close()

        # print('shape_from_file ' + str(data_shape_file))
        data_shape = [data_shape_file[1], data_shape_file[0]]

        # print('shape : ' + str(np.shape(x)))

        # totnumwp = data_shape[0] * data_shape[1] #num_wp + 1  # counting starts with zero

        plotdata_mat = np.asarray(plotdata_mat_lis)
        # print(str(plotdata_mat))
        # print('Number of gridpoints: ' + str(plotdata_mat.shape[0]))

        """
        Parameter Calculation
        """

        def param_calc(Inside_Vector, length_Inside_Vector, vectorName):
            alpha = []
            gamma = []
            rdist = []
            for itx in analyze_tx:
                rdist_vec = plotdata_mat[Inside_Vector[:], 0:2] - txpos[itx, 0:2]  # r_wp -r_txpos
                rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  # distance norm: |r_wp -r_txpos|
                rssdata = plotdata_mat[Inside_Vector[:], 2 + itx]  # rss-mean for each wp
                popt, pcov = curve_fit(rsm_model, rdist_temp, rssdata)
                del pcov

                alpha.append(popt[0])
                gamma.append(popt[1])
                # print('tx #' + str(itx+1) + ' alpha= ' + str(alpha[itx]) + ' gamma= ' + str(gamma[itx]))
                rdist.append(rdist_temp)
            print('\nVectors for convenient copy/paste for '+str(vectorName) )
            print('alpha = ' + str(alpha))
            print('gamma = ' + str(gamma))
            rdist_temp = np.reshape(rdist, [num_tx, length_Inside_Vector])
            return [alpha, gamma, rdist_temp]

        """
        Model fit
        """
        # Model selection
        model_groundtruth = False
        model_Inside = True
        model_Line_10 = False
        model_Line_5 = False
        model_Z_Vector_small = True
        Data_Set_Pool = 3
        # set Data_Set_Pool to a value 0-3
        # param 0: use all points
        # param 1: use only 10 points

        Inside_Vector = np.zeros((540,), dtype=np.int)
        counter = 0
        Inside_array = np.arange(549, 1464, 1)
        Groundtruth_Vector = np.arange(0, totnumwp, 1)
        Line_Vector_10 = np.arange(992, 1028, 1)
        Line_Vector_5 = np.arange(992, 1028, 8)
        Z_Vector_small = [999, 1304, 1609, 1444, 1021, 1326, 1631]
        for ntx in range(0, 15):
            for itx in range(16, 52):
                Inside_Vector[counter] = Inside_array[itx + 61 * ntx]
                counter = counter + 1

        length_Inside_Vector = len(Inside_Vector)
        length_Line_Vector_10 = len(Line_Vector_10)
        length_Line_Vector_5 = len(Line_Vector_5)
        length_Z_Vector_small = len(Z_Vector_small)

        Name_Groundtruth_Vector = 'Groundtruth_Vector'
        Name_Inside_Vector = 'Inside_Vector'
        Name_Line_Vector_10 = 'Long Line Vector'
        Name_Line_Vector_5 = 'Short Line Vector'
        Name_Z_Vector_small = 'Small Z Vector'

        def rsm_model(dist_rsm, alpha_rsm, gamma_rsm):
            """Range Sensor Model (RSM) structure."""
            return -20 * np.log10(dist_rsm) - alpha_rsm * dist_rsm - gamma_rsm  # rss in db






        if model_Inside == True:
            (alpha, gamma, rdist_temp) = param_calc(Inside_Vector, length_Inside_Vector, Name_Inside_Vector)

        if model_groundtruth == True:
            (alpha_Groundtruth, gamma_Groundtruth, rdist_temp_groundtruth) = param_calc(Groundtruth_Vector, totnumwp, Name_Groundtruth_Vector)

        if model_Line_10 == True:
            (alpha_line_10, gamma_line_10, rdist_temp_line_10 ) = param_calc(Line_Vector_10, length_Line_Vector_10,Name_Line_Vector_10)

        if model_Line_5 == True:
            (alpha_line_5, gamma_line_5, rdist_temp_line_5) = param_calc(Line_Vector_5, length_Line_Vector_5,Name_Line_Vector_5)

        if model_Z_Vector_small == True:
            (alpha_Z_Vector_small, gamma_Z_Vector_small, rdist_temp_Z_Vector_small) = param_calc(Z_Vector_small, length_Z_Vector_small,Name_Z_Vector_small)





        """
        Plots
        """

        See_Plots = True  # Set to True if you want to see the plots
        x = plotdata_mat[:, 0]
        y = plotdata_mat[:, 1]

        if See_Plots == True:
            plot_fig1 = True
            if plot_fig1:
                fig = plt.figure(1)
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos)
                    rss_mean = plotdata_mat[:, 2 + itx]
                    rss_var = plotdata_mat[:, 2 + num_tx + itx]

                    rss_mat_ones = np.ones(np.shape(wp_matx)) * (-200)  # set minimum value for not measured points
                    rss_full_vec = np.reshape(rss_mat_ones, (len(xpos) * len(ypos), 1))

                    measured_wp_list = np.reshape(measured_wp_list, (len(measured_wp_list), 1))
                    rss_mean = np.reshape(rss_mean, (len(rss_mean), 1))

                    rss_full_vec[measured_wp_list, 0] = rss_mean

                    rss_full_mat = np.reshape(rss_full_vec, data_shape)

                    # mask all points which were not measured
                    rss_full_mat = np.ma.array(rss_full_mat, mask=rss_full_mat < -199)

                    val_sequence = np.linspace(-100, -20, 80 / 5 + 1)

                    CS = ax.contour(wp_matx, wp_maty, rss_full_mat, val_sequence)
                    ax.clabel(CS, inline=0, fontsize=10)
                    for itx_plot in analyze_tx:
                        ax.plot(txpos[itx_plot - 1, 0], txpos[itx_plot - 1, 1], 'or')

                    ax.grid()
                    ax.set_xlabel('x [mm]')
                    ax.set_ylabel('y [mm]')
                    ax.axis('equal')
                    ax.set_title('RSS field for TX# ' + str(itx + 1))

            if Data_Set_Pool == 0:
                Data_Set = Inside_Vector
                RDIST = rdist_temp
            elif Data_Set_Pool == 1:
                Data_Set = Line_Vector_10
                RDIST = rdist_temp_line_10
            elif Data_Set_Pool == 2:
                Data_Set = Line_Vector_5
                RDIST = rdist_temp_line_5
            elif Data_Set_Pool == 3:
                Data_Set = Z_Vector_small
                RDIST = rdist_temp_Z_Vector_small

            plot_fig3 = True
            if plot_fig3:
                fig = plt.figure(3)
                # plt.ion()
                for itx_plot in analyze_tx:
                    plt.plot(txpos[itx_plot - 1, 0], txpos[itx_plot - 1, 1], 'or')

                plt.plot(plotdata_mat[Data_Set[:], 0], plotdata_mat[Data_Set[:], 1], 'b.')

                plt.xlabel('Distance in mm (belt-drive)')
                plt.ylabel('Distance in mm (spindle-drive)')
                plt.xlim(x0[0] - 10, xn[0] + 100)
                plt.ylim(x0[1] - 10, xn[1] + 100)
                plt.grid()

            plot_fig2 = True
            if plot_fig2:
                fig = plt.figure(2)
                for itx in analyze_tx:
                    if model_groundtruth == True:
                        #rss_mean = plotdata_mat[:, 2 + itx]
                        #rss_var = plotdata_mat[:, 2 + num_tx + itx]
                        #rdist = np.array(rdist_temp[itx, :], dtype=float)
                        rdata = np.linspace(np.min(rdist_temp_groundtruth), np.max(rdist_temp_groundtruth), num=1000)
                    else:
                        rdata = np.linspace(np.min(rdist_temp), np.max(rdist_temp), num=1000)

                    rss_mean = plotdata_mat[Data_Set[:], 2 + itx]
                    rss_var = plotdata_mat[Data_Set[:], 2 + num_tx + itx]
                    rdist = np.array(RDIST[itx, :], dtype=float)

                    rss_mean = np.array(rss_mean, dtype=float)
                    rss_var = np.array(rss_var, dtype=float)
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111
                    ax = fig.add_subplot(pos)
                    ax.errorbar(rdist, rss_mean, yerr=rss_var,
                                fmt='ro', ecolor='g', label='Original Data')


                    if model_Inside == True:
                        ax.plot(rdata, rsm_model(rdata, alpha[itx], gamma[itx]), label='Fitted Curve')
                    if model_groundtruth == True:
                        ax.plot(rdata, rsm_model(rdata, alpha_Groundtruth[itx], gamma_Groundtruth[itx]),
                                label='Fitted Groundtruth')
                    if model_Line_10 == True:
                        ax.plot(rdata, rsm_model(rdata, alpha_line_10[itx], gamma_line_10[itx]),
                                label=('Fitted Curve: Line with ' + str(length_Line_Vector_10) + 'WP'))
                    if model_Line_5 == True:
                        ax.plot(rdata, rsm_model(rdata, alpha_line_5[itx], gamma_line_5[itx]),
                                label=('Fitted Curve: Line with' + str(length_Line_Vector_5) + 'WP'))
                    if model_Z_Vector_small == True:
                        ax.plot(rdata, rsm_model(rdata, alpha_Z_Vector_small[itx], gamma_Z_Vector_small[itx]),
                                label=('Fitted Curve: Z with' + str(length_Z_Vector_small) + 'WP'))
                    ax.legend(loc='upper right')
                    ax.grid()
                    ax.set_ylim([-110, -10])
                    ax.set_xlabel('Distance [mm]')
                    ax.set_ylabel('RSS [dB]')
                    ax.set_title('RSM for TX# ' + str(itx + 1))

    plt.show()

    return alpha, gamma


analyze_measdata_from_file()


