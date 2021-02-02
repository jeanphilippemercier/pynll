/*
 * Copyright (C) 2021 Jean-Philippe Mercier <jpmercier@uquake.org, http://www.uquake.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */


/*   NLLoc_main_func

	Wrapper around the main function to do global search earthquake
	location in 3-D models. The purpose of this function is for simpler
	integration with Cython.

*/

/*-----------------------------------------------------------------------
Jean-Philippe Mercier from Anthony Lomax main function (all credit to him)
uQuake


-------------------------------------------------------------------------*/


/*



.........1.........2.........3.........4.........5.........6.........7.........8

*/

#include "GridLib.h"
#include "ran1/ran1.h"
#include "velmod.h"
#include "GridMemLib.h"
#include "calc_crust_corr.h"
#include "phaseloclist.h"
#include "otime_limit.h"
#include "NLLocLib.h"

#ifdef CUSTOM_ETH
#include "custom_eth/eth_functions.h"
#endif


// function declarations


/** program to perform global search event locations */

#ifdef CUSTOM_ETH
#define NARGS_MIN 3
#define ARG_DESC "<control file> <snap_pid> <snap_param_file>"
#else
#define NARGS_MIN 2
#define ARG_DESC "<control file>"
#endif

int NLLoc_func(char fn_control_main[MAXLINE])
{

	int istat;

	// run NLLoc
	istat = NLLoc(pid_main, fn_control_main, NULL, -1, NULL, -1, 0, 0, 0, NULL);

	return(istat);