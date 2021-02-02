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

#include util.h

int NLLoc_func(char fn_control_main[MAXLINE])