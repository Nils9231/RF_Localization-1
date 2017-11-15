def curve_plot(alpha, gamma, plot_num):

    fig = plt.figure(plot_num)

    for itx in analyze_tx:
        pos = 321 + itx
        if len(analyze_tx) == 1:
            pos = 111

        estimated_plotdata = plotdata_mat
        rdist_vec = plotdata_mat[:, 0:2] - txpos[itx, 0:2]
        rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  # distance norm: |r_wp -r_txpos|
        estimated_plotdata[:, 2 + itx] = rsm_model(rdist_temp, alpha[itx], gamma[itx])

        ax = fig.add_subplot(pos)
        rss_mean = estimated_plotdata[:, 2 + itx]
        rss_var = estimated_plotdata[:, 2 + num_tx + itx]

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

        plt.show()